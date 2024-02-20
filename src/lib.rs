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

#[derive(Debug)]
pub struct CudaKernel {
    module: CUmodule,
    function: CUfunction,
    link_state: CUlinkState,
}

impl CudaKernel {
    fn destroy(&mut self) {
        check_cuda!(cuModuleUnload(self.module));
        check_cuda!(cuLinkDestroy(self.link_state));
    }
}

pub struct Cuda {
    nvjpeg: nvjpeg::Decoder,
    cuda_kernels: HashMap<&'static str, CudaKernel>,
    cuda_device: CUdevice,
    cuda_device_props: cudaDeviceProp,
    cuda_context: CUcontext,

    cuda_stream: cudaStream_t,
}

impl Drop for Cuda {
    fn drop(&mut self) {
        check_cuda!(cudaStreamDestroy(self.cuda_stream));
        self.nvjpeg.destroy();
        self.cuda_kernels.values_mut().for_each(|k| k.destroy());
        check_cuda!(cuCtxDestroy_v2(self.cuda_context));
    }
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
            nvjpeg: nvjpeg::Decoder::new(),
            cuda_kernels: HashMap::new(),
            cuda_device,
            cuda_device_props,
            cuda_context,
            cuda_stream,
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
        let nvrtc_args = [ptx_ver.as_ptr(), enable_rdx.as_ptr()];
        let compile_result = unsafe { nvrtcCompileProgram(cuda_program, 2, nvrtc_args.as_ptr()) };

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

const pattern_generation_kernel: &str = stringify!(
    extern "C" __global__ void generate_address_pattern() {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    }
    extern "C" __global__ void generate_address_pattern() {
        const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    }
);

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
    fn test_cuda_rgb_to_rgba() {
        let mut test_cuda = Cuda::new();
        test_cuda.setup_kernels();

        const pitch: u32 = 4096_u32;
        const height: u32 = 3072_u32;
        const size: u32 = pitch * height;
        const rgb_byte_size: usize = size as usize * 3;
        const rgba_byte_size: usize = size as usize * 4;
        const pitch_addr: *const c_void = &pitch as *const u32 as *const c_void;
        const size_addr: *const c_void = &size as *const u32 as *const c_void;

        let mut image_device_addr: CUdeviceptr = 0;
        check_cuda!(cuMemAlloc_v2(&mut image_device_addr, rgba_byte_size));

        let mut test_pattern_image_vec = Vec::with_capacity(rgba_byte_size);
        test_pattern_image_vec.resize(rgba_byte_size, 0);
        generate_test_pattern_byte_increment(&mut test_pattern_image_vec, rgb_byte_size);
        check_cuda!(cuMemcpyHtoD_v2(
            image_device_addr,
            test_pattern_image_vec.as_ptr() as *const c_void,
            rgba_byte_size
        ));

        let mut args: [*mut c_void; 3] = [
            &mut image_device_addr as *mut u64 as *mut c_void,
            pitch_addr.cast_mut(),
            size_addr.cast_mut(),
        ];

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

        check_cuda!(cudaEventRecord(startTestEvent, test_cuda.cuda_stream));
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
        check_cuda!(cudaEventRecord(stopTestEvent, test_cuda.cuda_stream));

        check_cuda!(cudaEventSynchronize(stopTestEvent));
        check_cuda!(cudaEventElapsedTime(
            &mut testTime,
            startTestEvent,
            stopTestEvent
        ));

        dbg!(testTime);

        check_cuda!(cuCtxSynchronize());
        check_cuda!(cuStreamSynchronize(test_cuda.cuda_stream));
        check_cuda!(cudaDeviceSynchronize());

        let mut test_result_image_vec = Vec::with_capacity(rgba_byte_size);
        test_result_image_vec.resize(rgba_byte_size, 0_u8);
        check_cuda!(cuMemcpyDtoH_v2(
            test_result_image_vec.as_mut_ptr() as *mut c_void,
            image_device_addr,
            rgba_byte_size
        ));

        // TODO: move to Cuda
        check_cuda!(cuMemFree_v2(image_device_addr));

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

    #[test]
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
                stage_num += 1;
                test_cuda.launch_kernel(
                    "rgb_to_rgba",
                    4096,
                    3072,
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
