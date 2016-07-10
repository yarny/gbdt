#include "forest_py.h"

#include "datastore_py.h"
#include "gbdt_py_base.h"
#include "src/gbdt_algo/evaluation.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"

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
  if (!status.ok()) {
    ThrowException(status);
  }
  return scores;
}

void InitForestPy(py::module &m) {
  py::class_<ForestPy>(m, "Forest")
      .def(py::init<const string&>())
      .def("as_json", &ForestPy::ToJson)
      .def("predict", &ForestPy::Predict);
}
