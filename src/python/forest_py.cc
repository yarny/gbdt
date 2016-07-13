#include "forest_py.h"

#include "datastore_py.h"
#include "gbdt_py_base.h"
#include "src/gbdt_algo/evaluation.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"

using gbdt::ForestPy;

namespace gbdt {

ForestPy::ForestPy(const string& str) {
  Forest forest;
  auto status = JsonUtils::FromJson(str, &forest);
  if (!status.ok()) ThrowException(status);

  forest_ = std::move(forest);
}

string ForestPy::ToJson() const {
  string json_str;
  auto status = JsonUtils::ToJson(forest_, &json_str);
  if (!status.ok()) ThrowException(status);

  return json_str;
}

vector<double> ForestPy::Predict(DataStorePy* data_store_py) const {
  vector<double> scores;
  auto status = EvaluateForest(data_store_py->data_store(),
                               forest_,
                               &scores);
  if (!status.ok()) ThrowException(status);
  return scores;
}

void ForestPy::PredictAndOutput(DataStorePy* data_store_py,
                                const list<int>& test_points,
                                const string& output_dir) const {
  auto status = EvaluateForest(data_store_py->data_store(),
                               forest_,
                               test_points,
                               output_dir);
  if (!status.ok()) ThrowException(status);
}

}  // namespace gbdt

void InitForestPy(py::module &m) {
  py::class_<ForestPy>(m, "Forest")
      .def(py::init<const string&>())
      .def("as_json", &ForestPy::ToJson)
      .def("predict", &ForestPy::Predict)
      .def("predict_and_output", &ForestPy::PredictAndOutput);
}
