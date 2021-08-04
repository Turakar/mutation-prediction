import mutation_prediction.data.baseline as baseline


def assert_param(variant, dataset, descriptor, key, value):
    params = baseline.load_hyperparameters(variant)
    assert params[dataset][descriptor][key] == value


def test_load_hyperparameters_cnn():
    assert_param("CNN", "A", "VHSE", "batch_size", 40)
    assert_param("CNN", "B", "Identity", "learning_rate", 0.01)
    assert_param("CNN", "D", "PCscores", "model", "cnn_1")
    assert_param("CNN", "C", "zScales", "epoch", 250)


def test_load_hyperparameters_glmnet():
    assert_param("GLMNET", "A", "VHSE", "alpha", 0.3)
    assert_param("GLMNET", "B", "mutInd", "lambda", 4)


def test_load_hyperparameters_mlp():
    assert_param("MLP", "A", "VHSE", "batch_size", 20)
    assert_param("MLP", "B", "mutInd", "learning_rate", 0.01)
    assert_param("MLP", "C", "zScales", "init_scale", 0.01)
    assert_param("MLP", "D", "protVec", "model", "model_12")
    assert_param("MLP", "A", "sPairs", "epoch", 40)


def test_load_hyperparameters_rf():
    assert_param("RF", "A", "mutInd", "max_features", 0.33)
    assert_param("RF", "D", "sPairs", "max_features", "sqrt")
    assert_param("RF", "B", "sScales", "n_estimators", 200)


def test_load_hyperparameters_svm():
    assert_param("SVM", "C", "VHSE", "sigma", 1e-2)
    assert_param("SVM", "D", "mutInd", "C", 64)


def test_load_hyperparameters_xgb():
    assert_param("XGB", "A", "VHSE", "nrounds", 1500)
    assert_param("XGB", "B", "Identity", "max_depth", 15)
    assert_param("XGB", "C", "zScales", "eta", 0.05)
    assert_param("XGB", "D", "PCscores", "colsample_bytree", 0.3)


def test_load_scores():
    scores = baseline.load_scores()
    assert scores["A"]["Identity"]["CNN"]["RMSE"] == 22.674
    assert scores["B"]["sScales"]["GLMNET"]["rsq"] == 0.613
    assert scores["C"]["sPairs"]["RF"]["RMSE"] == 0.457
    assert scores["D"]["zScales"]["XGB"]["rsq"] == 0.386
