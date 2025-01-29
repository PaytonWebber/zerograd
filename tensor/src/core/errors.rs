#[derive(Debug)]
pub enum TensorError {
    BroadcastError(String),
    CreationError(String),
    MovementError(String),
    IndexError(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::CreationError(msg) => write!(f, "Creation error: {}", msg),
            TensorError::BroadcastError(msg) => write!(f, "Broadcast error: {}", msg),
            TensorError::MovementError(msg) => write!(f, "Movement error: {}", msg),
            TensorError::IndexError(msg) => write!(f, "Index error: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}
