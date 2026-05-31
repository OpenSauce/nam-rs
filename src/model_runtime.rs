//! Architecture-agnostic runtime: [`Model`] dispatches over the `.nam`'s declared
//! architecture so consumers run any supported model without branching.

use crate::error::Error;
use crate::lstm::Lstm;
use crate::model::{ModelConfig, NamModel};
use crate::wavenet::WaveNet;

/// A runnable NAM model of any supported architecture.
///
/// Build with [`Model::from_nam`]; then call [`Model::process_buffer`] on the audio
/// thread. `#[non_exhaustive]` so future architectures don't break downstream
/// `match`es.
#[non_exhaustive]
#[derive(Debug)]
pub enum Model {
    /// A WaveNet model.
    WaveNet(WaveNet),
    /// An LSTM model.
    Lstm(Lstm),
}

impl Model {
    /// Build the runtime matching `model.architecture`. All allocation happens here.
    pub fn from_nam(model: &NamModel) -> Result<Self, Error> {
        match &model.config {
            ModelConfig::WaveNet(_) => Ok(Model::WaveNet(WaveNet::new(model)?)),
            ModelConfig::Lstm(_) => Ok(Model::Lstm(Lstm::new(model)?)),
        }
    }

    /// Process a buffer of mono samples in place. Allocation-free.
    pub fn process_buffer(&mut self, io: &mut [f32]) {
        match self {
            Model::WaveNet(w) => w.process_buffer(io),
            Model::Lstm(l) => l.process_buffer(io),
        }
    }

    /// Process a single mono sample. Allocation-free.
    pub fn process_sample(&mut self, x: f32) -> f32 {
        match self {
            Model::WaveNet(w) => w.process_sample(x),
            Model::Lstm(l) => l.process_sample(x),
        }
    }

    /// Reset all internal state to the model's initial conditions.
    pub fn reset(&mut self) {
        match self {
            Model::WaveNet(w) => w.reset(),
            Model::Lstm(l) => l.reset(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_WAVENET: &str = r#"{
        "version": "0.5.4", "architecture": "WaveNet",
        "config": { "layers": [{
            "input_size": 1, "condition_size": 1, "channels": 1, "head_size": 1,
            "kernel_size": 1, "dilations": [1], "activation": "ReLU",
            "gated": false, "head_bias": false
        }], "head": null, "head_scale": 10.0 },
        "weights": [1.0, 2.0, 0.5, 1.0, 3.0, 0.1, 0.5, 10.0]
    }"#;

    const TINY_LSTM: &str = r#"{
        "version": "0.5.4", "architecture": "LSTM",
        "config": { "input_size": 1, "hidden_size": 1, "num_layers": 1 },
        "weights": [1.0,0.0, 0.0,0.0, 2.0,0.0, 0.0,0.0, 0.0,0.0,0.0,0.0, 0.0, 0.0, 3.0, 0.5]
    }"#;

    #[test]
    fn from_nam_builds_wavenet() {
        let m = NamModel::from_json_str(TINY_WAVENET).unwrap();
        let mut model = Model::from_nam(&m).unwrap();
        assert!(matches!(model, Model::WaveNet(_)));
        let mut buf = [0.5_f32];
        model.process_buffer(&mut buf);
        assert!((buf[0] - 10.0).abs() < 1e-5, "got {}", buf[0]);
    }

    #[test]
    fn from_nam_builds_lstm() {
        let m = NamModel::from_json_str(TINY_LSTM).unwrap();
        let mut model = Model::from_nam(&m).unwrap();
        assert!(matches!(model, Model::Lstm(_)));
        let mut buf = [0.5_f32];
        model.process_buffer(&mut buf);
        assert!((buf[0] - 1.1623).abs() < 1e-3, "got {}", buf[0]);
    }
}
