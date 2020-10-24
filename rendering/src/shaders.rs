pub mod simple_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/simple.vert"
    }
}

pub mod simple_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/simple.frag"
    }
}
