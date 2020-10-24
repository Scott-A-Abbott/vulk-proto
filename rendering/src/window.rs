use {
    cgmath::*,
    std::sync::Arc,
    vulkano::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        command_buffer::{AutoCommandBufferBuilder, DynamicState},
        descriptor::{
            descriptor_set::FixedSizeDescriptorSetsPool,
            pipeline_layout::PipelineLayoutAbstract,
        },
        device::{Device, DeviceExtensions, Queue},
        framebuffer::{
            Framebuffer, FramebufferAbstract, RenderPass, RenderPassAbstract, RenderPassDesc,
            Subpass,
        },
        image::{swapchain::SwapchainImage, ImageUsage},
        instance::{Instance, PhysicalDevice},
        pipeline::{vertex::SingleBufferDefinition, viewport::Viewport, GraphicsPipeline},
        swapchain::{
            self, AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface,
            SurfaceTransform, Swapchain, SwapchainCreationError,
        },
        sync::{self, FlushError, GpuFuture},
    },
    vulkano_win::VkSurfaceBuild,
    winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window as Win, WindowBuilder},
    },
};

const PARTIAL: f32 = 1.57 / 1.85;
const VERT_WIDTH: f32 = PARTIAL * PARTIAL;

const VERTS: [Vertex; 4] = [
    Vertex {
        position: [-VERT_WIDTH, -PARTIAL, 0.0],
    },
    Vertex {
        position: [-VERT_WIDTH, 0.0, 0.0],
    },
    Vertex {
        position: [0.0, -PARTIAL, 0.0],
    },
    Vertex {
        position: [0.0, 0.0, 0.0],
    },
];
const VERTS2: [Vertex; 4] = [
    Vertex {
        position: [0.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.0, PARTIAL, 0.0],
    },
    Vertex {
        position: [VERT_WIDTH, 0.0, 0.0],
    },
    Vertex {
        position: [VERT_WIDTH, PARTIAL, 0.0],
    },
];

struct Entity {
    verts: [Vertex; 4],
    transform: Vector3<f32>
}

impl Entity {
    fn model(&self) -> Matrix4<f32> {
        Matrix4::from_translation(self.transform.clone())
    }
}

type VertexBuffer = Arc<CpuAccessibleBuffer<[Vertex]>>;
type WindowImages = Vec<Arc<SwapchainImage<Win>>>;

//make into a builder pattern struct that returns an event loop and renderer
struct RenderFactory {}

pub struct Renderer {
    _instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface<Win>>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Win>>,
    images: WindowImages,
}
impl Renderer {
    pub fn new() -> (Self, EventLoop<()>) {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(None, &required_extensions, None).unwrap();
        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        log::info!(
            "Using device: {} (type: {:?})",
            physical.name(),
            physical.ty()
        );

        let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        let (device, mut queues) = Device::new(
            physical,
            physical.supported_features(),
            &device_ext,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .unwrap();

        log::info!("Number of queues: {}", queues.len());

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            let dimensions: [u32; 2] = surface.window().inner_size().into();
            Swapchain::new(
                device.clone(),
                surface.clone(),
                caps.min_image_count,
                format,
                dimensions,
                1,
                ImageUsage::color_attachment(),
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
                FullscreenExclusive::Default,
                true,
                ColorSpace::SrgbNonLinear,
            )
            .unwrap()
        };

        let self_ = Self {
            _instance: instance,
            device,
            surface,
            queue,
            swapchain,
            images,
        };

        (self_, event_loop)
    }

    fn create_render_pass(&self) -> Arc<RenderPass<impl RenderPassDesc>> {
        Arc::new(
            vulkano::single_pass_renderpass!(
                self.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: self.swapchain.format(),
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )
            .unwrap(),
        )
    }

    fn create_pipeline<D>(
        &self,
        render_pass: Arc<RenderPass<D>>,
    ) -> Arc<
        GraphicsPipeline<
            SingleBufferDefinition<Vertex>,
            Box<dyn PipelineLayoutAbstract + Send + Sync>,
            Arc<RenderPass<D>>,
        >,
    >
    where
        D: RenderPassDesc,
    {
        use crate::shaders::{simple_fs, simple_vs};
        let vs = simple_vs::Shader::load(self.device.clone()).unwrap();
        let fs = simple_fs::Shader::load(self.device.clone()).unwrap();

        Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                // .triangle_list()
                .triangle_strip()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(self.device.clone())
                .unwrap(),
        )
    }

    pub fn run(&mut self, event_loop: EventLoop<()>) {
        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };
        let queue = self.queue.clone();

        let entities = {
            let e1 = Entity {
                verts: VERTS,
                transform: Vector3::new(0.0, 0.0, 0.0),
            };

            let e2 = Entity {
                verts: VERTS2,
                transform: Vector3::new(0.0, 0.0, 0.0)
            };

            vec![e1, e2]
        };

        let mut vertex_buffer = VertexBufferBuilder::new(self.device.clone());
        for e in entities.iter() {
            vertex_buffer.with_verts(&e.verts);
        }
        let vertex_buffer = vertex_buffer.build();

        let render_pass = self.create_render_pass();
        let mut framebuffers =
            window_size_dependent_setup(&self.images, render_pass.clone(), &mut dynamic_state);
        let pipeline = self.create_pipeline(render_pass.clone());

        // log::info!("{:?}", pipeline.layout());
        // let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
        // let mut pool = FixedSizeDescriptorSetsPool::new(layout.clone());

        // let _descriptor_set = pool.next();

        let mut recreate_swapchain = false;
        let mut previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        let surface = self.surface.clone();
        let mut swapchain = self.swapchain.clone();
        let device = self.device.clone();

        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    winit::event::KeyboardInput {
                        state: winit::event::ElementState::Pressed,
                        virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => {}
                },
                WindowEvent::Resized(_) => recreate_swapchain = true,
                _ => {}
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut dynamic_state,
                    );
                    recreate_swapchain = false;
                }
                let (image_num, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                if suboptimal {
                    recreate_swapchain = true;
                }
                let clear_values = vec![[0.7, 0.8, 0.9, 1.0].into()];
                let mut builder = AutoCommandBufferBuilder::primary_one_time_submit(
                    device.clone(),
                    queue.family(),
                )
                .unwrap();

                builder
                    .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
                    .unwrap()
                    .draw(
                        pipeline.clone(),
                        &dynamic_state,
                        vertex_buffer.clone(),
                        (),
                        (),
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = builder.build().unwrap();
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        })
    }
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position);

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Win>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    images
        .iter()
        .map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

struct VertexBufferBuilder {
    device: Arc<Device>,
    verts: Vec<Vertex>,
}
impl VertexBufferBuilder {
    fn new(device: Arc<Device>) -> Self {
        Self {
            device: device.clone(),
            verts: Default::default(),
        }
    }

    fn with_verts(&mut self, verts: &[Vertex]) -> &mut Self {
        self.verts.extend_from_slice(verts);
        self
    }

    fn build(&self) -> VertexBuffer {
        CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::all(),
            false,
            self.verts.iter().cloned(),
        )
        .unwrap()
    }
}