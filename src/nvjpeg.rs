use super::*;

macro_rules! check_nvjpeg {
    ($($nvjpegCommandArgs:tt)*) => {
        let status = unsafe { $($nvjpegCommandArgs)* };
        if status != nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
            panic!("{} failed. status = {}", stringify!($($nvjpegCommandArgs)*), status);
        }
    };
}

pub struct Decoder {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    decoder: nvjpegJpegDecoder_t,
    decode_params: nvjpegDecodeParams_t,
    decoder_state: nvjpegJpegState_t,
    host_buffers: [nvjpegBufferPinned_t; 2],
    device_buffer: nvjpegBufferDevice_t,
    streams: [nvjpegJpegStream_t; 2],
}

#[derive(Debug)]
pub struct ImageInfo {
    pub components: i32,
    pub subsampling: nvjpegChromaSubsampling_t,
    pub width: [i32; NVJPEG_MAX_COMPONENT as usize],
    pub height: [i32; NVJPEG_MAX_COMPONENT as usize],
}

#[derive(Debug)]
pub struct ExtImage {
    pub cuda_ext_mem: cudaExternalMemory_t,
    pub nvjpeg_image_output: nvjpegImage_t,
}

impl Decoder {
    pub fn destroy(&mut self) {
        check_nvjpeg!(nvjpegDecodeParamsDestroy(self.decode_params));
        check_nvjpeg!(nvjpegJpegStreamDestroy(self.streams[0]));
        check_nvjpeg!(nvjpegJpegStreamDestroy(self.streams[1]));
        check_nvjpeg!(nvjpegBufferPinnedDestroy(self.host_buffers[0]));
        check_nvjpeg!(nvjpegBufferPinnedDestroy(self.host_buffers[1]));
        check_nvjpeg!(nvjpegBufferDeviceDestroy(self.device_buffer));
        check_nvjpeg!(nvjpegJpegStateDestroy(self.decoder_state));
        check_nvjpeg!(nvjpegDecoderDestroy(self.decoder));
        check_nvjpeg!(nvjpegJpegStateDestroy(self.state));
        check_nvjpeg!(nvjpegDestroy(self.handle));
    }
    pub fn new() -> Self {
        let mut dev_allocator = nvjpegDevAllocator_t {
            dev_malloc: Some(cudaMalloc),
            dev_free: Some(cudaFree),
        };
        let mut pinned_allocator = nvjpegPinnedAllocator_t {
            pinned_malloc: Some(cudaHostAlloc),
            pinned_free: Some(cudaFreeHost),
        };

        let mut handle: nvjpegHandle_t = std::ptr::null_mut();
        let mut nvjpeg_status = unsafe {
            nvjpegCreateEx(
                nvjpegBackend_t_NVJPEG_BACKEND_HARDWARE,
                &mut dev_allocator,
                &mut pinned_allocator,
                NVJPEG_FLAGS_DEFAULT,
                &mut handle,
            )
        };
        if nvjpeg_status == nvjpegStatus_t_NVJPEG_STATUS_ARCH_MISMATCH {
            println!("no hardware nvjpeg backend");
            nvjpeg_status = unsafe {
                nvjpegCreateEx(
                    nvjpegBackend_t_NVJPEG_BACKEND_DEFAULT,
                    &mut dev_allocator,
                    &mut pinned_allocator,
                    NVJPEG_FLAGS_DEFAULT,
                    &mut handle,
                )
            };
        }
        if nvjpeg_status != nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
            println!("failed to create nvjpeg backend");
        }

        let mut state: nvjpegJpegState_t = std::ptr::null_mut();
        check_nvjpeg!(nvjpegJpegStateCreate(handle, &mut state));

        let mut decoder: nvjpegJpegDecoder_t = std::ptr::null_mut();
        check_nvjpeg!(nvjpegDecoderCreate(
            handle,
            nvjpegBackend_t_NVJPEG_BACKEND_DEFAULT,
            &mut decoder,
        ));

        let mut decoder_state: nvjpegJpegState_t = std::ptr::null_mut();
        check_nvjpeg!(nvjpegDecoderStateCreate(
            handle,
            decoder,
            &mut decoder_state
        ));

        let mut host_buffers: [nvjpegBufferPinned_t; 2] =
            [std::ptr::null_mut(), std::ptr::null_mut()];
        check_nvjpeg!(nvjpegBufferPinnedCreate(
            handle,
            std::ptr::null_mut(),
            &mut host_buffers[0],
        ));
        check_nvjpeg!(nvjpegBufferPinnedCreate(
            handle,
            std::ptr::null_mut(),
            &mut host_buffers[1],
        ));

        let mut device_buffer: nvjpegBufferDevice_t = std::ptr::null_mut();
        check_nvjpeg!(nvjpegBufferDeviceCreate(
            handle,
            std::ptr::null_mut(),
            &mut device_buffer,
        ));

        let mut streams: [nvjpegJpegStream_t; 2] =
            [std::ptr::null_mut(), std::ptr::null_mut()];
        check_nvjpeg!(nvjpegJpegStreamCreate(
            handle,
            &mut streams[0]
        ));
        check_nvjpeg!(nvjpegJpegStreamCreate(
            handle,
            &mut streams[1]
        ));

        let mut decode_params: nvjpegDecodeParams_t = std::ptr::null_mut();
        check_nvjpeg!(nvjpegDecodeParamsCreate(
            handle,
            &mut decode_params
        ));

        Self {
            handle,
            state,
            decoder,
            decode_params,
            decoder_state,
            host_buffers,
            device_buffer,
            streams,
        }
    }

    pub fn get_image_info(&self, jpeg_buffer: &[u8]) -> ImageInfo {
        let mut components = 0;
        let mut subsampling: nvjpegChromaSubsampling_t = 0;
        let mut width = [0; NVJPEG_MAX_COMPONENT as usize];
        let mut height = [0; NVJPEG_MAX_COMPONENT as usize];
        check_nvjpeg!(nvjpegGetImageInfo(
            self.handle,
            jpeg_buffer.as_ptr(),
            jpeg_buffer.len(),
            &mut components,
            &mut subsampling,
            &mut width as *mut i32,
            &mut height as *mut i32,
        ));
        ImageInfo {
            components,
            subsampling,
            width,
            height,
        }
    }

    pub fn get_image_extents(&self, jpeg_buffer: &[u8]) -> [u32; 2] {
        let image_info = self.get_image_info(jpeg_buffer);
        [image_info.width[0] as u32, image_info.height[1] as u32]
    }

    pub fn decode_image(&self, jpeg_buffer: &[u8], ext_image_out: &mut ExtImage, cuda_stream: cudaStream_t) {
        check_cuda!(cudaStreamSynchronize(cuda_stream));

        check_nvjpeg!(nvjpegStateAttachDeviceBuffer(
            self.decoder_state,
            self.device_buffer
        ));

        check_nvjpeg!(nvjpegDecodeParamsSetOutputFormat(
            self.decode_params,
            nvjpegOutputFormat_t_NVJPEG_OUTPUT_BGRI
        ));

        check_nvjpeg!(nvjpegJpegStreamParse(
            self.handle,
            jpeg_buffer.as_ptr(),
            jpeg_buffer.len(),
            0,
            0,
            self.streams[0]
        ));

        check_nvjpeg!(nvjpegStateAttachPinnedBuffer(
            self.decoder_state,
            self.host_buffers[0]
        ));

        check_nvjpeg!(nvjpegDecodeJpegHost(
            self.handle,
            self.decoder,
            self.decoder_state,
            self.decode_params,
            self.streams[0]
        ));

        check_cuda!(cudaStreamSynchronize(cuda_stream));

        check_nvjpeg!(nvjpegDecodeJpegTransferToDevice(
            self.handle,
            self.decoder,
            self.decoder_state,
            self.streams[0],
            cuda_stream
        ));

        check_nvjpeg!(nvjpegDecodeJpegDevice(
            self.handle,
            self.decoder,
            self.decoder_state,
            &mut ext_image_out.nvjpeg_image_output,
            cuda_stream
        ));
    }
}
