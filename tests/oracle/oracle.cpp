// tests/oracle/oracle.cpp
// Usage: oracle <model.nam> <input.json> <output.json>
// Reads input.json (a flat JSON array of numbers), runs the model with a fresh
// Reset and NO prewarm (zero-init streaming, matching nam-rs), writes output.json
// (a flat JSON array). NAM_SAMPLE is double by default in NAMCore.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "NAM/get_dsp.h"

// Parse a flat JSON array of numbers ("[1.0, -2, 3e-1]") into doubles.
// Tolerant: treats any of "[],\n\r\t " as separators between number tokens.
static std::vector<double> parse_json_array(const std::string& text)
{
  std::vector<double> out;
  std::string tok;
  auto flush = [&]() {
    if (!tok.empty()) { out.push_back(std::stod(tok)); tok.clear(); }
  };
  for (char c : text)
  {
    if (c == '[' || c == ']' || c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t')
      flush();
    else
      tok.push_back(c);
  }
  flush();
  return out;
}

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::fprintf(stderr, "usage: %s <model.nam> <input.json> <output.json>\n", argv[0]);
    return 2;
  }

  // Load input.
  std::ifstream in(argv[2]);
  if (!in) { std::fprintf(stderr, "cannot open input %s\n", argv[2]); return 1; }
  std::stringstream ss; ss << in.rdbuf();
  std::vector<double> input = parse_json_array(ss.str());
  const int n = static_cast<int>(input.size());

  // Load model.
  std::unique_ptr<nam::DSP> model;
  try { model = nam::get_dsp(std::filesystem::path(argv[1])); }
  catch (const std::exception& e) { std::fprintf(stderr, "get_dsp failed: %s\n", e.what()); return 1; }
  if (!model) { std::fprintf(stderr, "get_dsp returned null\n"); return 1; }

  // Zero-init streaming pass: fresh Reset sized to the whole buffer, NO prewarm.
  std::vector<double> output(static_cast<size_t>(n), 0.0);
  model->Reset(model->GetExpectedSampleRate(), n > 0 ? n : 1);
  if (n > 0)
  {
    double* inPtr = input.data();
    double* outPtr = output.data();
    model->process(&inPtr, &outPtr, n);
  }

  // Write output as a flat JSON array of f32-precision values (the crate runs f32).
  std::ofstream of(argv[3]);
  if (!of) { std::fprintf(stderr, "cannot open output %s\n", argv[3]); return 1; }
  of << "[";
  for (int i = 0; i < n; ++i)
  {
    if (i) of << ", ";
    of << static_cast<float>(output[static_cast<size_t>(i)]);
  }
  of << "]";
  return 0;
}
