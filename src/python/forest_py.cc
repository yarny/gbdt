#include "forest_py.h"

#include "datastore_py.h"
#include "gbdt_py_base.h"
#include "src/gbdt_algo/evaluation.h"
#include "src/proto/tree.pb.h"
#include "src/utils/json_utils.h"

ForestPy::ForestPy(const string& str) {
  Forest forest;
  if (!JsonUtils::FromJson(str, &forest)) {
    ThrowException(Status(gbdt::error::INVALID_ARGUMENT,
                          "Failed to parse Json."));

  }
  forest_ = std::move(forest);
}

string ForestPy::ToJson() const {
  return JsonUtils::ToJsonOrDie(forest_);
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
