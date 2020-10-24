fn main() {
    pretty_env_logger::init_custom_env("LOG");
    let (mut window, event_loop) = rendering::Renderer::new();
    window.run(event_loop);
}