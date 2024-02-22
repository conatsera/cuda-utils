#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;
use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    os::raw::c_void,
};

mod nvjpeg;

macro_rules! check_nvrtc {
    ($($nvrtcCommandArgs:tt)*) => {
        let status = unsafe { $($nvrtcCommandArgs)* };
        if status != nvrtcResult_NVRTC_SUCCESS {
            panic!("{} failed. status = {}", stringify!($($nvrtcCommandArgs)*), status);
        }
    };
}

#[macro_export]
macro_rules! check_cuda {
    ($($cudaCommandArgs:tt)*) => {
        let status = unsafe { $($cudaCommandArgs)* };
        if status != cudaError_cudaSuccess {
            panic!("{} failed. status = {}", stringify!($($cudaCommandArgs)*), status);
        }
    };
}

macro_rules! check_cuda_no_panic {
    ($($cudaCommandArgs:tt)*) => {
        let status = unsafe { $($cudaCommandArgs)* };
        if status != cudaError_cudaSuccess {
            println!("{} failed. status = {}", stringify!($($cudaCommandArgs)*), status);
        }
    };
}

pub struct CudaKernel {
    module: CUmodule,
    function: CUfunction,
    link_state: CUlinkState,
}

impl CudaKernel {
    fn destroy(&mut self) {
        check_cuda_no_panic!(cuModuleUnload(self.module));
        check_cuda_no_panic!(cuLinkDestroy(self.link_state));
    }
}

pub struct Cuda {
    nvjpeg: nvjpeg::Decoder,
    cuda_kernels: HashMap<&'static str, CudaKernel>,
    allocations: Vec<CUdeviceptr_v2>,
    cached_allocs: HashMap<&'static *const c_void, CUdeviceptr_v2>,
    _cuda_device: CUdevice,
    cuda_device_props: cudaDeviceProp,
    cuda_context: CUcontext,

    cuda_stream: cudaStream_t,

    block_move_sequence_buffers: [(u32, u32, [BlockMove; 64]); 5],
}

impl Drop for Cuda {
    fn drop(&mut self) {
        check_cuda_no_panic!(cudaStreamDestroy(self.cuda_stream));
        self.allocations.iter().for_each(|dptr| { check_cuda_no_panic!(cuMemFree_v2(*dptr)); });
        self.cuda_kernels.values_mut().for_each(|k| k.destroy());
        self.nvjpeg.destroy();
        check_cuda_no_panic!(cuCtxDestroy_v2(self.cuda_context));
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockMove {
    size: u32,
    block_size: u32,
    position: u32,
}

impl BlockMove {
    const fn default() -> Self {
        BlockMove {
            size: 0,
            block_size: 0,
            position: 0,
        }
    }
}

const fn calculate_optimal_block_size(size: u32) -> u32 {
    let pow2_divisor = size.trailing_zeros();
    if pow2_divisor > 10 {
        return 1024;
    }
    if pow2_divisor != 10 {
        let mut i = 1023;
        while i > 0 {
            if size % i == 0 {
                return i;
            }
            i-=1;
        }
    }
    1_u32 << pow2_divisor
}

const fn generate_free_segment_block_moves(size: u32) -> (u32, u32, [BlockMove; 64]) {
    let mut block_move_sequence = [BlockMove::default(); 64];
    let mut free_segment_size = size / 4;
    let mut position = size - free_segment_size;
    let mut sequence_num = 0;
    while free_segment_size > 0 && sequence_num < 64 {
        block_move_sequence[sequence_num as usize] = BlockMove{
            size: free_segment_size,
            block_size: calculate_optimal_block_size(free_segment_size),
            position
        };
        sequence_num+=1;
        free_segment_size /= 4;
        free_segment_size *= 3;
        position -= free_segment_size;
    }
    (sequence_num, position, block_move_sequence)
}

const DEFAULT_IMAGE_SIZES: [usize; 5] = [
    4096*3072,
    2048*1536,
    3840*2160,
    2560*1440,
    1920*1080
];

const fn generate_builtin_block_sizes() -> [(u32, u32, [BlockMove; 64]); 5] {
    let mut block_moves = [(0, 0, [BlockMove::default(); 64]); 5];
    let mut size_count = 0;
    while size_count < 5 {
        block_moves[size_count] = generate_free_segment_block_moves(DEFAULT_IMAGE_SIZES[size_count] as u32);
        size_count+=1;
    }
    block_moves
}

impl Cuda {
    pub fn new() -> Self {
        check_cuda!(cuInit(0));
        let mut cuda_device: CUdevice = 0;
        check_cuda!(cudaGetDevice(&mut cuda_device));
        let mut cuda_device_props: cudaDeviceProp = unsafe { std::mem::zeroed() };
        check_cuda!(cudaGetDeviceProperties_v2(
            &mut cuda_device_props,
            cuda_device
        ));
        let mut cuda_context: CUcontext = std::ptr::null_mut();
        check_cuda!(cuCtxCreate_v2(&mut cuda_context, 0, cuda_device));
        let mut cuda_stream: cudaStream_t = std::ptr::null_mut();
        check_cuda!(cudaStreamCreateWithFlags(
            &mut cuda_stream,
            cudaStreamNonBlocking
        ));
        
        Self {
            allocations: Vec::new(),
            nvjpeg: nvjpeg::Decoder::new(),
            cuda_kernels: HashMap::new(),
            cached_allocs: HashMap::new(),
            _cuda_device: cuda_device,
            cuda_device_props,
            cuda_context,
            cuda_stream,
            block_move_sequence_buffers: generate_builtin_block_sizes(),
        }
    }

    pub fn allocate(&mut self, device_addr: &mut CUdeviceptr, byte_size: usize) {
        check_cuda!(cuMemAlloc_v2(device_addr, byte_size));
        self.allocations.push(*device_addr);
    }

    pub fn create_global_buffer(&mut self, host_buffer: *const c_void, byte_size: usize) -> CUdeviceptr {
        // cursed caching based on host_buffer address. This might not be a good idea but it works
        if let Some(device_addr) = self.cached_allocs.get(&host_buffer) {
            *device_addr
        } else {
            let mut device_addr: CUdeviceptr = 0;
            self.allocate(&mut device_addr, byte_size);
            check_cuda!(cuMemcpyHtoD_v2(
                device_addr,
                host_buffer,
                byte_size
            ));
            device_addr
        }
    }

    pub fn setup_kernels(&mut self) {
        self.create_kernel("rgb_to_rgba", rgb_to_rgba_kernel);
    }

    pub fn get_ptx_version(&self) -> i32 {
        self.cuda_device_props.major
    }

    pub fn get_jpeg_image_extents(&self, jpeg_buffer: &[u8]) -> [u32; 2] {
        self.nvjpeg.get_image_extents(jpeg_buffer)
    }

    pub fn destroy_image(&self, ext_image_out: &mut nvjpeg::ExtImage) {
        for i in 0..NVJPEG_MAX_COMPONENT {
            check_cuda!(cudaFree(
                ext_image_out.nvjpeg_image_output.channel[i as usize].cast()
            ));
        }

        if ext_image_out.cuda_ext_mem != std::ptr::null_mut() {
            check_cuda!(cudaDestroyExternalMemory(ext_image_out.cuda_ext_mem));
        }
    }

    pub fn create_vk_image(
        &self,
        ext_image_fd: &std::fs::File,
        image_dimensions: [u32; 2],
    ) -> nvjpeg::ExtImage {
        let image_size = image_dimensions[0] as u64 * image_dimensions[1] as u64 * 4;

        let cudaExtMemBufferDesc: cudaExternalMemoryBufferDesc = cudaExternalMemoryBufferDesc {
            offset: 0,
            size: image_size,
            flags: 0,
        };
        #[cfg(windows)]
        let mut cudaExtMemHandleDesc = cudaExternalMemoryHandleDesc {
            type_: cudaExternalMemoryHandleType_cudaExternalMemoryHandleTypeOpaqueWin32,
            handle: cudaExternalMemoryHandleDesc__bindgen_ty_1 {
                win32: cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1 {
                    handle: ext_image_fd.as_raw_handle(),
                    name: std::ptr::null_mut(),
                },
            },
            size: image_size,
            flags: 0,
        };

        let mut image_output_device_ext_mem: cudaExternalMemory_t = std::ptr::null_mut();
        check_cuda!(cudaImportExternalMemory(
            &mut image_output_device_ext_mem,
            &mut cudaExtMemHandleDesc
        ));

        let mut image_channel_buffer = std::ptr::null_mut();
        check_cuda!(cudaExternalMemoryGetMappedBuffer(
            &mut image_channel_buffer,
            image_output_device_ext_mem,
            &cudaExtMemBufferDesc
        ));

        nvjpeg::ExtImage {
            cuda_ext_mem: image_output_device_ext_mem,
            nvjpeg_image_output: nvjpegImage_t {
                channel: [
                    image_channel_buffer.cast(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ],
                pitch: [image_dimensions[0] as usize * 3, 0, 0, 0],
            },
        }
    }

    pub fn create_cuda_image(&self, image_dimensions: [u32; 2]) -> nvjpeg::ExtImage {
        let image_size = image_dimensions[0] as u64 * image_dimensions[1] as u64 * 4;

        let mut image_channel_buffer = std::ptr::null_mut();
        check_cuda!(cudaMalloc(&mut image_channel_buffer, image_size as usize));

        nvjpeg::ExtImage {
            cuda_ext_mem: std::ptr::null_mut(),
            nvjpeg_image_output: nvjpegImage_t {
                channel: [
                    image_channel_buffer.cast(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ],
                pitch: [image_dimensions[0] as usize * 3, 0, 0, 0],
            },
        }
    }

    pub fn rgb_to_rgba_cuda(&mut self, size: u32, image_device_addr: *mut c_void) {
        let image_size_index = DEFAULT_IMAGE_SIZES.into_iter().position(|s| s == size as usize).unwrap();
        let block_moves = self.block_move_sequence_buffers[image_size_index];
        let mut block_moves_device = self.create_global_buffer(
            &block_moves.2 as *const BlockMove as *mut c_void,
            block_moves.0 as usize * std::mem::size_of::<BlockMove>()
        );
        
        let mut args: [*mut c_void; 4] = [
            image_device_addr,
            &self.block_move_sequence_buffers[image_size_index].0 as *const u32 as *const c_void as *mut c_void,
            &self.block_move_sequence_buffers[image_size_index].1 as *const u32 as *const c_void as *mut c_void,
            &mut block_moves_device as *mut u64 as *mut c_void,
        ];
        self.launch_kernel(
            "rgb_to_rgba",
            1,
            1,
            1,
            1,
            1,
            1,
            &mut args as *mut *mut c_void,
        );
    }

    pub fn decode_image(&self, jpeg_buffer: &[u8], ext_image_out: &mut nvjpeg::ExtImage) {
        self.nvjpeg
            .decode_image(jpeg_buffer, ext_image_out, self.cuda_stream);
    }

    #[inline]
    pub fn launch_kernel(
        &self,
        kernel_name: &'static str,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        kernel_params: *mut *mut c_void,
    ) {
        let kernel = self.cuda_kernels.get(kernel_name).unwrap();
        check_cuda!(cuLaunchKernel(
            kernel.function,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            0,
            self.cuda_stream,
            kernel_params,
            std::ptr::null_mut()
        ));
    }

    pub fn create_kernel(&mut self, kernel_name: &'static str, cuda_code_str: &str) {
        let mut cuda_program: nvrtcProgram = std::ptr::null_mut();
        let kernel_name_cstr = CString::new(kernel_name).unwrap();
        let kernel_code = CString::new(cuda_code_str).unwrap();
        check_nvrtc!(nvrtcCreateProgram(
            &mut cuda_program,
            kernel_code.as_ptr(),
            kernel_name_cstr.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null()
        ));

        let ptx_ver = CString::new(format!("-arch=sm_{}0", self.get_ptx_version())).unwrap();
        let enable_rdx = CString::new("-rdc=true").unwrap();
        let reg_mem_usage_report = CString::new("--ptxas-options=-v").unwrap();
        let nvrtc_args = [ptx_ver.as_ptr(), enable_rdx.as_ptr(), reg_mem_usage_report.as_ptr()];
        let compile_result = unsafe { nvrtcCompileProgram(cuda_program, 3, nvrtc_args.as_ptr()) };

        let mut log_size: usize = 0;
        check_nvrtc!(nvrtcGetProgramLogSize(cuda_program, &mut log_size));

        if log_size > 1 {
            let mut log_string_bytes = Vec::with_capacity(log_size);
            log_string_bytes.resize(log_size, 0);
            check_nvrtc!(nvrtcGetProgramLog(
                cuda_program,
                log_string_bytes.as_mut_ptr()
            ));
            let log_string = unsafe { CStr::from_ptr(log_string_bytes.as_ptr()) };
            dbg!(log_string, compile_result);
        }

        let mut ptx_size: usize = 0;
        check_nvrtc!(nvrtcGetPTXSize(cuda_program, &mut ptx_size));

        let mut ptx_bytes = Vec::with_capacity(ptx_size);
        ptx_bytes.resize(ptx_size, 0);
        check_nvrtc!(nvrtcGetPTX(cuda_program, ptx_bytes.as_mut_ptr()));

        check_nvrtc!(nvrtcDestroyProgram(&mut cuda_program));

        if compile_result != 0 {
            return;
        }

        let mut cuda_link_state: CUlinkState = std::ptr::null_mut();
        check_cuda!(cuLinkCreate_v2(
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut cuda_link_state
        ));

        let cuda_path = std::env::var("CUDA_PATH").unwrap();
        let cudadevrt_lib_path =
            CString::new(format!("{cuda_path}/lib/x64/cudadevrt.lib")).unwrap();
        check_cuda!(cuLinkAddFile_v2(
            cuda_link_state,
            CUjitInputType_enum_CU_JIT_INPUT_LIBRARY,
            cudadevrt_lib_path.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut()
        ));

        check_cuda!(cuLinkAddData_v2(
            cuda_link_state,
            CUjitInputType_enum_CU_JIT_INPUT_PTX,
            ptx_bytes.as_mut_ptr() as *mut c_void,
            ptx_size,
            kernel_name_cstr.as_ptr(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut()
        ));

        let mut code_size: usize = 0;
        let mut code_addr = std::ptr::null_mut();
        check_nvrtc!(cuLinkComplete(
            cuda_link_state,
            &mut code_addr,
            &mut code_size
        ));

        let mut module: CUmodule = std::ptr::null_mut();
        check_cuda!(cuModuleLoadData(&mut module, code_addr));
        let mut function: CUfunction = std::ptr::null_mut();
        check_cuda!(cuModuleGetFunction(
            &mut function,
            module,
            kernel_name_cstr.as_ptr()
        ));

        self.cuda_kernels.insert(
            kernel_name,
            CudaKernel {
                module,
                function,
                link_state: cuda_link_state,
            },
        );
    }
}

const rgb_to_rgba_kernel: &str = include_str!("../kernels/rgb_to_rgba.cu");

#[repr(C)]
pub struct RgbToRgbaArgs {
    image: *mut u8,
    pitch: u32,
    size: u32,
    stage: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_pattern_byte_increment(pattern_buffer: &mut [u8], pattern_len: usize) {
        for (pos, pattern_byte) in pattern_buffer.iter_mut().enumerate() {
            if pos < pattern_len {
                *pattern_byte = (pos % 0xFF) as u8;
            }
        }
    }


    #[test]
    fn test_pattern_cuda_rgb_to_rgba() {
        let mut test_cuda = Cuda::new();
        test_cuda.setup_kernels();

        const width: u32 = 4096_u32;
        const height: u32 = 3072_u32;
        const size: u32 = width * height;
        const rgb_byte_size: usize = size as usize * 3;
        const rgba_byte_size: usize = size as usize * 4;

        let mut image_device_addr: CUdeviceptr = 0;
        test_cuda.allocate(&mut image_device_addr, rgba_byte_size);

        let mut test_pattern_image_vec = Vec::with_capacity(rgba_byte_size);
        test_pattern_image_vec.resize(rgba_byte_size, 0);
        generate_test_pattern_byte_increment(&mut test_pattern_image_vec, rgb_byte_size);
        let test_pattern_image_vec = test_pattern_image_vec;
        check_cuda!(cuMemcpyHtoD_v2(
            image_device_addr,
            test_pattern_image_vec.as_ptr() as *const c_void,
            rgba_byte_size
        ));

        let mut startTestEvent: cudaEvent_t = std::ptr::null_mut();
        let mut stopTestEvent: cudaEvent_t = std::ptr::null_mut();
        let mut testTime = 0_f32;

        check_cuda!(cudaEventCreateWithFlags(
            &mut startTestEvent,
            cudaEventBlockingSync
        ));
        check_cuda!(cudaEventCreateWithFlags(
            &mut stopTestEvent,
            cudaEventBlockingSync
        ));

        for _ in 0..30 {
            check_cuda!(cuMemcpyHtoD_v2(
                image_device_addr,
                test_pattern_image_vec.as_ptr() as *const c_void,
                rgba_byte_size
            ));
            check_cuda!(cudaEventRecord(startTestEvent, test_cuda.cuda_stream));
            test_cuda.rgb_to_rgba_cuda(size, &mut image_device_addr as *mut u64 as *mut c_void);
            check_cuda!(cudaEventRecord(stopTestEvent, test_cuda.cuda_stream));

            check_cuda!(cudaEventSynchronize(stopTestEvent));
            check_cuda!(cudaEventElapsedTime(
                &mut testTime,
                startTestEvent,
                stopTestEvent
            ));
            dbg!(testTime);
        
            let mut test_result_image_vec = Vec::with_capacity(rgba_byte_size);
            test_result_image_vec.resize(rgba_byte_size, 0_u8);
            check_cuda!(cuMemcpyDtoH_v2(
                test_result_image_vec.as_mut_ptr() as *mut c_void,
                image_device_addr,
                rgba_byte_size
            ));

            let mut index = rgb_byte_size;
            for (pos, byte) in test_result_image_vec.into_iter().enumerate().rev() {
                //println!("{:x} {:x} {:x} {:x} {:x}", pos, rgba_byte_size - 1 - pos, byte, index - 1, *test_pattern_image_vec.get(index - 1).unwrap());
                if pos % 4 == 3 {
                    assert_eq!(byte, 0, "pattern mismatch @ {}", pos / 4);
                } else {
                    index -= 1;
                    assert_eq!(
                        byte,
                        *test_pattern_image_vec.get(index).unwrap(),
                        "pattern mismatch @ {}, pattern_index: {}",
                        pos / 4,
                        index
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "needs generated/test assets"]
    fn test_cuda_jpeg_file_decode() {
        let mut test_cuda = Cuda::new();
        test_cuda.setup_kernels();
        let jpeg_file = std::fs::read("../test.jpg").unwrap();

        let mut image_out = test_cuda.create_cuda_image([4096, 3072]);

        const width: u32 = 4096_u32;
        const size: u32 = width * 3072_u32;
        const width_addr: *const c_void = &width as *const u32 as *const c_void;
        const size_addr: *const c_void = &size as *const u32 as *const c_void;
        let mut stage_num = 1_u8;
        let stage_num_addr: *mut c_void = &mut stage_num as *mut u8 as *mut c_void;

        let image_out_addr = &mut image_out.nvjpeg_image_output.channel[0] as *mut *mut u8;

        let mut args: [*mut c_void; 4] = [
            image_out_addr.cast(),
            width_addr.cast_mut(),
            size_addr.cast_mut(),
            stage_num_addr,
        ];

        dbg!(&image_out);
        dbg!(args);

        let mut startTestEvent: cudaEvent_t = std::ptr::null_mut();
        let mut stopTestEvent: cudaEvent_t = std::ptr::null_mut();
        let mut testTime = 0_f32;

        check_cuda!(cudaEventCreateWithFlags(
            &mut startTestEvent,
            cudaEventBlockingSync
        ));
        check_cuda!(cudaEventCreateWithFlags(
            &mut stopTestEvent,
            cudaEventBlockingSync
        ));

        check_cuda!(cudaEventRecord(startTestEvent, std::ptr::null_mut()));

        for _ in 0..30 {
            test_cuda.decode_image(&jpeg_file, &mut image_out);
            let mut startEvent: cudaEvent_t = std::ptr::null_mut();
            let mut stopEvent: cudaEvent_t = std::ptr::null_mut();
            let mut loopTime = 0_f32;

            check_cuda!(cudaEventCreateWithFlags(
                &mut startEvent,
                cudaEventBlockingSync
            ));
            check_cuda!(cudaEventCreateWithFlags(
                &mut stopEvent,
                cudaEventBlockingSync
            ));

            check_cuda!(cudaEventRecord(startEvent, std::ptr::null_mut()));
            while stage_num >= 4 {
                test_cuda.launch_kernel(
                    "rgb_to_rgba",
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    &mut args as *mut *mut c_void,
                );
            }
            check_cuda!(cudaEventRecord(stopEvent, std::ptr::null_mut()));

            check_cuda!(cudaEventSynchronize(stopEvent));
            check_cuda!(cudaEventElapsedTime(&mut loopTime, startEvent, stopEvent));

            dbg!(loopTime);
        }

        check_cuda!(cudaEventRecord(stopTestEvent, std::ptr::null_mut()));
        check_cuda!(cudaEventSynchronize(stopTestEvent));
        check_cuda!(cudaEventElapsedTime(
            &mut testTime,
            startTestEvent,
            stopTestEvent
        ));

        dbg!(testTime);

        test_cuda.destroy_image(&mut image_out);
    }
}
