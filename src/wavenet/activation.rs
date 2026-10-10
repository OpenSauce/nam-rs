//! Pointwise activation functions used by WaveNet layers and the gating module.
//!
//! Ported from NeuralAmpModelerCore `NAM/activations.{h,cpp}`. Shared by
//! `layer.rs` (the post-conv activation) and `gating.rs` (primary/secondary).

use crate::error::Error;

// Rational approximation of tanh:
// f(x) = x(A + A|x| + (B + C|x|)x^2) / (D + (D + x^2)|x + E·x|x||)
// Coefficients are curve-fitted; not standard constants.
const FAST_TANH_A: f32 = 2.45550750702956;
const FAST_TANH_B: f32 = 0.893229853513558;
const FAST_TANH_C: f32 = 0.821226666969744;
const FAST_TANH_D: f32 = 2.44506634652299;
const FAST_TANH_E: f32 = 0.814642734961073;

/// Pointwise activation applied after the dilated conv + mix-in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Activation {
    Tanh,
    Relu,
    Sigmoid,
    /// LeakyReLU with the given negative slope (`x > 0 ? x : slope*x`).
    LeakyRelu(f32),
    Hardtanh,
    Softsign,
    SiLU, // aka swish
    Hardswish,
    Fasttanh,
}

impl Activation {
    pub(super) fn from_spec(spec: &crate::model::ActivationSpec) -> Result<Self, Error> {
        use crate::model::ActivationSpec;
        match spec {
            ActivationSpec::Named {
                name,
                negative_slope,
            } => match name.as_str() {
                "Tanh" => Ok(Self::Tanh),
                "ReLU" => Ok(Self::Relu),
                "Sigmoid" => Ok(Self::Sigmoid),
                "LeakyReLU" => Ok(Self::LeakyRelu(negative_slope.unwrap_or(0.01))),
                "Hardtanh" => Ok(Self::Hardtanh),
                "Softsign" => Ok(Self::Softsign),
                "SiLU" => Ok(Self::SiLU),
                "Hardswish" => Ok(Self::Hardswish),
                "Fasttanh" => Ok(Self::Fasttanh),
                other => Err(Error::UnsupportedActivation(other.to_string())),
            },
            ActivationSpec::Unsupported(v) => {
                Err(Error::UnsupportedFeature(format!("activation: {v}")))
            }
        }
    }

    #[inline]
    pub(super) fn apply(self, x: f32) -> f32 {
        match self {
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => sigmoid(x),
            Self::LeakyRelu(slope) => {
                if x > 0.0 {
                    x
                } else {
                    slope * x
                }
            }
            Self::Hardtanh => hard_tanh(x),
            Self::Softsign => softsign(x),
            Self::SiLU => swish(x),
            Self::Hardswish => hardswish(x),
            Self::Fasttanh => fast_tanh(x),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn hard_tanh(x: f32) -> f32 {
    let t = if x < -1.0 { -1.0 } else { x };
    if t > 1.0 {
        1.0
    } else {
        t
    }
}

#[inline]
fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

#[inline]
fn swish(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn hardswish(x: f32) -> f32 {
    let t = x + 3.0;
    let clamped = if t < 0.0 {
        0.0
    } else if t > 6.0 {
        6.0
    } else {
        t
    };
    x * clamped * (1.0 / 6.0)
}

#[inline]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;

    return (x * (FAST_TANH_A + FAST_TANH_A * ax + (FAST_TANH_B + FAST_TANH_C * ax) * x2))
        / (FAST_TANH_D + (FAST_TANH_D + x2) * (x + FAST_TANH_E * x * ax).abs());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_spec_resolves_named_activations() {
        use crate::model::ActivationSpec;
        let named = |n: &str| ActivationSpec::Named {
            name: n.into(),
            negative_slope: None,
        };
        assert_eq!(
            Activation::from_spec(&named("Tanh")).unwrap(),
            Activation::Tanh
        );
        assert_eq!(
            Activation::from_spec(&named("ReLU")).unwrap(),
            Activation::Relu
        );
        assert_eq!(
            Activation::from_spec(&named("Sigmoid")).unwrap(),
            Activation::Sigmoid
        );
        assert_eq!(
            Activation::from_spec(&named("LeakyReLU")).unwrap(),
            Activation::LeakyRelu(0.01)
        );
        assert_eq!(
            Activation::from_spec(&ActivationSpec::Named {
                name: "LeakyReLU".into(),
                negative_slope: Some(0.2)
            })
            .unwrap(),
            Activation::LeakyRelu(0.2)
        );
        assert_eq!(
            Activation::from_spec(&named("Hardtanh")).unwrap(),
            Activation::Hardtanh
        );
        assert_eq!(
            Activation::from_spec(&named("Softsign")).unwrap(),
            Activation::Softsign
        );
        assert_eq!(
            Activation::from_spec(&named("SiLU")).unwrap(),
            Activation::SiLU
        );
        assert_eq!(
            Activation::from_spec(&named("Hardswish")).unwrap(),
            Activation::Hardswish
        );
        assert_eq!(
            Activation::from_spec(&named("Fasttanh")).unwrap(),
            Activation::Fasttanh
        );
    }

    #[test]
    fn from_spec_rejects_unknown_and_unsupported() {
        use crate::model::ActivationSpec;
        let bad_name = ActivationSpec::Named {
            name: "NotAnActivation".into(),
            negative_slope: None,
        };
        assert!(matches!(
            Activation::from_spec(&bad_name),
            Err(crate::Error::UnsupportedActivation(_))
        ));
        let list = ActivationSpec::Unsupported(serde_json::json!(["ReLU", "Tanh"]));
        assert!(matches!(
            Activation::from_spec(&list),
            Err(crate::Error::UnsupportedFeature(_))
        ));
    }

    #[test]
    fn leaky_relu_applies_slope() {
        let a = Activation::LeakyRelu(0.01);
        assert_eq!(a.apply(2.0), 2.0);
        assert!((a.apply(-2.0) - (-0.02)).abs() < 1e-9);
        assert_eq!(a.apply(0.0), 0.0);
    }

    #[test]
    fn sigmoid_matches_reference() {
        // sigmoid(0) = 0.5 exactly; pin the formula gating relies on.
        assert_eq!(Activation::Sigmoid.apply(0.0), 0.5);
        let want = 1.0_f32 / (1.0 + (-1.5_f32).exp());
        assert!((Activation::Sigmoid.apply(1.5) - want).abs() < 1e-9);
    }

    #[test]
    fn hardtanh_clamps_to_unit_range() {
        let a = Activation::Hardtanh;
        assert_eq!(a.apply(-5.0), -1.0);
        assert_eq!(a.apply(-1.0), -1.0);
        assert_eq!(a.apply(-0.25), -0.25);
        assert_eq!(a.apply(0.0), 0.0);
        assert_eq!(a.apply(0.75), 0.75);
        assert_eq!(a.apply(1.0), 1.0);
        assert_eq!(a.apply(5.0), 1.0);
    }

    #[test]
    fn softsign_matches_reference() {
        let a = Activation::Softsign;
        assert_eq!(a.apply(0.0), 0.0);
        assert_eq!(a.apply(1.0), 0.5);
        assert_eq!(a.apply(-3.0), -0.75);
        // Odd, and bounded by (-1, 1) even for large inputs.
        assert_eq!(a.apply(-2.5), -a.apply(2.5));
        assert!(a.apply(1e6) < 1.0 && a.apply(-1e6) > -1.0);
    }

    #[test]
    fn silu_matches_reference() {
        let a = Activation::SiLU;
        assert_eq!(a.apply(0.0), 0.0);
        let want = 1.5_f32 / (1.0 + (-1.5_f32).exp());
        assert!((a.apply(1.5) - want).abs() < 1e-6);
        let want = -2.0_f32 / (1.0 + 2.0_f32.exp());
        assert!((a.apply(-2.0) - want).abs() < 1e-6);
        // Tends to identity for large positive x and to 0 for large negative x.
        assert!((a.apply(20.0) - 20.0).abs() < 1e-5);
        assert!(a.apply(-20.0).abs() < 1e-5);
    }

    #[test]
    fn hardswish_matches_reference() {
        let a = Activation::Hardswish;
        // x <= -3 → 0, x >= 3 → x, otherwise x * (x + 3) / 6.
        assert_eq!(a.apply(-5.0), 0.0);
        assert_eq!(a.apply(-3.0), 0.0);
        assert_eq!(a.apply(0.0), 0.0);
        assert_eq!(a.apply(3.0), 3.0);
        assert_eq!(a.apply(5.0), 5.0);
        assert!((a.apply(1.0) - 4.0 / 6.0).abs() < 1e-6);
        assert!((a.apply(-1.0) - (-2.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn fasttanh_approximates_tanh() {
        let a = Activation::Fasttanh;
        assert_eq!(a.apply(0.0), 0.0);
        // Max abs error of the rational approximation is ~4.4e-4 on [-10, 10].
        for i in -1000..=1000 {
            let x = i as f32 * 0.01;
            let got = a.apply(x);
            assert!(
                (got - x.tanh()).abs() < 5e-4,
                "fast_tanh({x}) = {got}, tanh = {}",
                x.tanh()
            );
            assert_eq!(a.apply(-x), -got, "not odd at x = {x}");
        }
    }
}
