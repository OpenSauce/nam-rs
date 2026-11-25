//! Pointwise activation functions used by WaveNet layers and the gating module.
//!
//! Ported from NeuralAmpModelerCore `NAM/activations.{h,cpp}`. Shared by
//! `layer.rs` (the post-conv activation) and `gating.rs` (primary/secondary).

use crate::error::Error;

/// Pointwise activation applied after the dilated conv + mix-in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Activation {
    Tanh,
    Relu,
    Sigmoid,
    /// LeakyReLU with the given negative slope (`x > 0 ? x : slope*x`).
    LeakyRelu(f32),
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
            Self::Tanh => tanh(x),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => sigmoid(x),
            Self::LeakyRelu(slope) => {
                if x > 0.0 {
                    x
                } else {
                    slope * x
                }
            }
        }
    }

    /// [`Self::apply`] over a slice. The match is outside the loop, so the
    /// loop vectorizes.
    pub(super) fn apply_slice(self, xs: &mut [f32]) {
        self.apply_slice_with(xs);
    }

    /// The body of [`Self::apply_slice`], inlined into each build.
    #[inline(always)]
    pub(super) fn apply_slice_with(self, xs: &mut [f32]) {
        match self {
            Self::Tanh => xs.iter_mut().for_each(|x| *x = tanh(*x)),
            Self::Relu => xs.iter_mut().for_each(|x| *x = x.max(0.0)),
            Self::Sigmoid => xs.iter_mut().for_each(|x| *x = sigmoid(*x)),
            Self::LeakyRelu(slope) => xs
                .iter_mut()
                .for_each(|x| *x = if *x > 0.0 { *x } else { slope * *x }),
        }
    }
}

/// `e^y = s * (1 + p)` for `y` in `[-80, 80]`, branch-free so it vectorizes.
///
/// `y = n ln 2 + r` with `|r| <= ln 2 / 2`; `s = 2^n` is built from its
/// exponent bits and `p = e^r - 1` is the Taylor series to `r^7` (error
/// below 1e-8). `ln 2` is split into a short high part and a low part (Cody
/// and Waite's reduction), so `n * LN2_HI` is exact and `r` keeps its
/// precision.
#[inline(always)]
fn exp_parts(y: f32) -> (f32, f32) {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    // 355/512 has 9 significant bits; LN2_LO is the rest of ln 2.
    const LN2_HI: f32 = 355.0 / 512.0;
    const LN2_LO: f32 = -2.121_944_4e-4;
    // 1.5 * 2^23: adding it rounds to an integer, kept in the low bits.
    const ROUND: f32 = 12_582_912.0;
    let y = y.clamp(-80.0, 80.0);
    let shifted = y * LOG2_E + ROUND;
    let n = shifted - ROUND;
    let r = (y - n * LN2_HI) - n * LN2_LO;
    let n_bits = shifted.to_bits().wrapping_sub(ROUND.to_bits());
    let s = f32::from_bits(n_bits.wrapping_add(127) << 23);
    let mut p = 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    (s, p * r)
}

/// `tanh(x) = (e^2x - 1) / (e^2x + 1)`, through [`exp_parts`], so unlike the
/// libm call it vectorizes. `e^2x - 1` is formed as `(s - 1) + s * p`, which
/// keeps small `x` accurate. About 1e-7 from the exact value, as libm's
/// `tanhf`. The input is clamped at `|x| = 9`, where `tanh` is within 6e-8
/// of 1. NaN stays NaN.
#[inline(always)]
pub(crate) fn tanh(x: f32) -> f32 {
    let a = x.abs();
    let a = if a > 9.0 { 9.0 } else { a };
    let (s, p) = exp_parts(2.0 * a);
    let expm1 = (s - 1.0) + s * p;
    (expm1 / (expm1 + 2.0)).copysign(x)
}

/// The logistic sigmoid `1 / (1 + e^-x)`, through [`exp_parts`] so it
/// vectorizes too.
#[inline(always)]
pub(crate) fn sigmoid(x: f32) -> f32 {
    let (s, p) = exp_parts(-x);
    1.0 / (1.0 + (s + s * p))
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
    }

    #[test]
    fn from_spec_rejects_unknown_and_unsupported() {
        use crate::model::ActivationSpec;
        let bad_name = ActivationSpec::Named {
            name: "Softsign".into(),
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
        // sigmoid(0) = 0.5 exactly; the rest within f32 rounding.
        assert_eq!(Activation::Sigmoid.apply(0.0), 0.5);
        let want = 1.0_f32 / (1.0 + (-1.5_f32).exp());
        assert!((Activation::Sigmoid.apply(1.5) - want).abs() < 1e-6);
    }

    /// The largest error of `tanh` and `sigmoid` against the f64 functions,
    /// over every 1e-4 step of `[-12, 12]`. libm's own `tanhf` is about 1e-7
    /// off.
    #[test]
    fn fast_tanh_and_sigmoid_stay_close_to_f64() {
        let (mut tanh_err, mut sigmoid_err) = (0.0_f64, 0.0_f64);
        for step in -120_000..=120_000 {
            let x = step as f32 * 1e-4;
            let xf = x as f64;
            tanh_err = tanh_err.max((tanh(x) as f64 - xf.tanh()).abs());
            sigmoid_err = sigmoid_err.max((sigmoid(x) as f64 - 1.0 / (1.0 + (-xf).exp())).abs());
        }
        assert!(tanh_err < 3e-7, "tanh error {tanh_err:e}");
        assert!(sigmoid_err < 3e-7, "sigmoid error {sigmoid_err:e}");
    }

    #[test]
    fn fast_tanh_is_odd_bounded_and_exact_at_zero() {
        assert_eq!(tanh(0.0), 0.0);
        for x in [1e-6_f32, 0.3, 1.0, 4.0, 9.0, 1e6, f32::INFINITY] {
            assert_eq!(tanh(-x), -tanh(x), "odd at {x}");
            assert!(tanh(x).abs() <= 1.0, "bounded at {x}");
        }
        assert!((tanh(f32::INFINITY) - 1.0).abs() < 1e-6);
        assert!(tanh(f32::NAN).is_nan());
        assert!(sigmoid(f32::NAN).is_nan());
        // Small inputs keep their relative precision.
        for x in [1e-7_f32, 1e-5, 1e-3] {
            assert!((tanh(x) / x - 1.0).abs() < 1e-5, "relative at {x}");
        }
    }

    #[test]
    fn apply_slice_matches_apply() {
        let xs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
        for act in [
            Activation::Tanh,
            Activation::Relu,
            Activation::Sigmoid,
            Activation::LeakyRelu(0.1),
        ] {
            let mut got = xs.clone();
            act.apply_slice(&mut got);
            for (&x, &g) in xs.iter().zip(&got) {
                assert_eq!(g, act.apply(x), "{act:?} at {x}");
            }
        }
    }
}
