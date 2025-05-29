# Machine Learning Model for Acquire Valued Shoppers Challenge

Here's a structured approach to building your model:

## Phase 1: Data Understanding and Preparation

1.  **Load Data:**
    *   Load `transactions.csv`, `offers.csv`, `trainHistory.csv`, and `testHistory.csv` into Spark DataFrames.
    *   **Key Action:** You'll need to join these datasets. `trainHistory` (and `testHistory`) will be your base, enriched with `offers.csv` and aggregated data from `transactions.csv`.

    ```python
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("ShoppersChallenge").getOrCreate()

    transactions_df = spark.read.csv(
        "path/to/transactions.csv", header=True, inferSchema=True
    )
    offers_df = spark.read.csv(
        "path/to/offers.csv", header=True, inferSchema=True
    )
    train_history_df = spark.read.csv(
        "path/to/trainHistory.csv", header=True, inferSchema=True
    )
    test_history_df = spark.read.csv(
        "path/to/testHistory.csv", header=True, inferSchema=True
    )

    # Basic inspection
    print("Train History Schema:")
    train_history_df.printSchema()
    print("Train History Sample:")
    train_history_df.show(5)
    ```

2.  **Define the Target Variable:**
    *   The `repeater` column in `trainHistory.csv` is your target.
    *   Ensure it's numerical (0 or 1). For Spark ML, it's common to rename this to `label` and cast it to `double`.

    ```python
    from pyspark.sql.functions import col

    train_history_df = train_history_df.withColumn(
        "label", col("repeater").cast("double")
    )
    # test_history_df will not have 'repeater' or 'label'
    ```

## Phase 2: Feature Engineering (CRITICAL)

This is the most crucial phase. Create features that capture customer behavior, offer characteristics, and their interactions.

*   **Important:** Feature engineering should occur *before* detailed correlation analysis on raw columns.

**Examples of Features to Engineer:**

*   **Offer-Specific Features (from `offers_df` & `transactions_df`):**
    *   `offer_value`, `offer_quantity`
    *   Offer popularity (total times bought).
    *   Average spend on this offer.
    *   Number of unique customers who bought this offer.
*   **Customer-Specific Features (from `transactions_df`, aggregated per `id`):**
    *   Total customer spend.
    *   Number of distinct items bought by the customer.
    *   Number of shopping trips.
    *   Average basket size.
    *   RFM (Recency, Frequency, Monetary) scores.
*   **Customer-Offer Interaction Features:**
    *   How many times has *this customer* bought *this offer* before `offerdate`?
    *   How many times has *this customer* bought items from the *same category/brand/company* as *this offer*?
    *   Time since the customer last purchased *this offer*.
*   **Time-Based Features:**
    *   Day of the week, month of `offerdate`.
    *   Days since the customer's first transaction.

**Process for Feature Engineering:**

1.  Join `train_history_df` with `offers_df` (on `offer`).
2.  Aggregate `transactions_df` to create customer-level and offer-level features.
    *   **Crucial:** Features must only use transaction data *before* the `offerdate` for each row in `train_history_df` to prevent data leakage.
3.  Join these aggregated features back to your main training DataFrame (`train_df`).
4.  Ensure all engineered features are numerical or can be transformed into numerical representations.

```python
# --- This is a conceptual placeholder for your feature engineering ---
# --- You will need to implement detailed logic based on the ideas above ---
from pyspark.sql.functions import lit # For example purposes

# 1. Join train_history with offers
train_df = train_history_df.join(offers_df, "offer")

# 2. Example: Create placeholder engineered features
#    Replace these with your actual feature calculation logic.
#    Ensure your target variable is 'label'.
train_df = train_df.withColumn("num_past_purchases_offer", lit(0.0)) # Placeholder
train_df = train_df.withColumn("avg_spend_customer", lit(0.0))     # Placeholder
train_df = train_df.withColumn("offer_category_engineered", lit("CAT_A")) # Placeholder

# Select only necessary columns for the model, including 'label' and engineered features
# For example:
# feature_columns_for_model = ["num_past_purchases_offer", "avg_spend_customer", "offer_category_engineered"]
# train_df_processed = train_df.select(["id", "label"] + feature_columns_for_model)

print("DataFrame after basic feature engineering (schema):")
# train_df_processed.printSchema() # Uncomment when train_df_processed is fully defined
print("DataFrame after basic feature engineering (sample):")
# train_df_processed.show(5, truncate=False) # Uncomment
```

**Note:** Feature engineering is iterative. You'll likely revisit this after initial model evaluations.

## Phase 3: Correlation Analysis

*   Calculate correlations *after* you have engineered your numerical features.
    *   **Correlation with Target:** Identify features most correlated with `label`.
    *   **Inter-Feature Correlation:** Check for multicollinearity.

```python
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, IntegerType, LongType

# --- This step assumes 'train_df_processed' is populated with engineered numerical features ---
# --- and a 'label' column. ---

# Example: (Uncomment and adapt when train_df_processed and features are ready)
# numerical_feature_cols = [
#     f.name for f in train_df_processed.schema.fields
#     if isinstance(f.dataType, (DoubleType, IntegerType, LongType))
#     and f.name not in ['id', 'label'] # Exclude ID and label itself
# ]

# if numerical_feature_cols: # Proceed if there are numerical features
#     # Add 'label' to calculate its correlation with features
#     cols_for_corr_matrix = numerical_feature_cols + ["label"]
#     assembler_corr = VectorAssembler(
#         inputCols=cols_for_corr_matrix,
#         outputCol="corr_features_vec",
#         handleInvalid="skip"
#     )
#     # Ensure train_df_processed is not empty before transform
#     # temp_df_for_corr = train_df_processed.select(cols_for_corr_matrix).na.drop() # Handle NaNs before assembling
#     # if temp_df_for_corr.count() > 0:
#     #    df_vector_corr = assembler_corr.transform(temp_df_for_corr).select("corr_features_vec")
#     #    if df_vector_corr.count() > 0:
#     #        correlation_matrix = Correlation.corr(df_vector_corr, "corr_features_vec").head()
#     #        if correlation_matrix:
#     #            print("Pearson correlation matrix (condensed for label):")
#     #            corr_array = correlation_matrix[0].toArray()
#     #            label_correlations = corr_array[:-1, -1] # Correlation of each feature with the last col (label)
#     #            corr_with_label_df = spark.createDataFrame(
#     #                zip(numerical_feature_cols, label_correlations.tolist()),
#     #                ["feature", "correlation_with_label"]
#     #            )
#     #            corr_with_label_df.orderBy(col("correlation_with_label").desc()).show(truncate=False)
#     #        else:
#     #            print("Could not compute correlation matrix (empty result).")
#     #    else:
#     #        print("DataFrame for correlation is empty after assembling vectors.")
#     # else:
#     #    print("DataFrame for correlation is empty after selecting columns and dropping NaNs.")
# else:
#     print("No numerical features found for correlation analysis.")
```

## Phase 4: Data Splitting

*   Split your engineered and processed `train_df_processed` into training and validation sets.

```python
# --- This assumes 'train_df_processed' is ready ---
# (engineered_train_data, engineered_validation_data) = train_df_processed.randomSplit(
#     [0.8, 0.2], seed=42
# )
# print(f"Training data count: {engineered_train_data.count()}")
# print(f"Validation data count: {engineered_validation_data.count()}")
```

## Phase 5: ML Pipeline Configuration

Set up your pipeline with `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, and `LinearSVC`.

```python
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml.classification import LinearSVC
from pyspark.ml import Pipeline

# --- Define these based on your actual engineered features from Phase 2 ---
# --- These names must exist in 'engineered_train_data' and 'engineered_validation_data' ---
# Example:
categorical_cols_for_pipeline = ["offer_category_engineered"] # From your feature engineering
numerical_cols_for_pipeline = ["num_past_purchases_offer", "avg_spend_customer"] # From your feature engineering

# Stage 1: StringIndexer for categorical features
indexers = [
    StringIndexer(
        inputCol=col_name, outputCol=col_name + "_indexed", handleInvalid="keep"
    )
    for col_name in categorical_cols_for_pipeline
]

# Stage 2: OneHotEncoder for indexed categorical features
encoder_input_cols = [col_name + "_indexed" for col_name in categorical_cols_for_pipeline]
encoders = [
    OneHotEncoder(inputCol=idx_col, outputCol=idx_col + "_encoded")
    for idx_col in encoder_input_cols
]

# Stage 3: VectorAssembler to combine all features
# (OHE outputs + numerical features)
assembler_input_cols = [
    enc_col + "_encoded" for enc_col in encoder_input_cols
] + numerical_cols_for_pipeline
vector_assembler = VectorAssembler(
    inputCols=assembler_input_cols, outputCol="features", handleInvalid="keep" # or "skip"
)

# Stage 4: ML Estimator (LinearSVC)
lsvc = LinearSVC(labelCol="label", featuresCol="features", maxIter=10, regParam=0.1)

# Create the full pipeline
pipeline_stages = indexers + encoders + [vector_assembler, lsvc]
pipeline = Pipeline(stages=pipeline_stages)

print("Pipeline stages defined.")
```
**Notes:**
*   `handleInvalid="keep"` in `StringIndexer` and `VectorAssembler` can help manage unseen values or NaNs.
*   Ensure `labelCol` in `LinearSVC` matches your target column name (e.g., "label").
*   `featuresCol` in `LinearSVC` must match `outputCol` of `VectorAssembler`.

## Phase 6: Model Training

Fit the pipeline to your training data.

```python
# --- This assumes 'engineered_train_data' is your DataFrame after ---
# --- feature engineering and splitting, containing all necessary columns. ---

# print("Starting pipeline fitting...")
# # Ensure engineered_train_data is cached if it's going to be used multiple times (e.g. in CV)
# # engineered_train_data.cache()
# model = pipeline.fit(engineered_train_data)
# print("Pipeline fitting completed.")
# # engineered_train_data.unpersist() # Unpersist after use if cached
```

## Phase 7: Model Evaluation

Make predictions on your validation set and evaluate using AUC.

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --- This assumes 'model' is trained and 'engineered_validation_data' is ready ---

# print("Making predictions on validation data...")
# predictions_validation = model.transform(engineered_validation_data)
# print("Predictions made.")

# # It's good to inspect some predictions
# # predictions_validation.select("id", "label", "rawPrediction", "prediction").show(5)

# evaluator_auc = BinaryClassificationEvaluator(
#     labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
# )
# auc_validation = evaluator_auc.evaluate(predictions_validation)
# print(f"Area Under ROC Curve (AUC) on validation data: {auc_validation}")
```
*Note: `LinearSVC` outputs `rawPrediction`. Use this for AUC.*

## Phase 8: Iteration and Hyperparameter Tuning ("Melhor Solução")

To find the "best solution":

1.  **Iterate on Feature Engineering:** This often yields the largest improvements.
2.  **Hyperparameter Tuning:** Use `CrossValidator` and `ParamGridBuilder` for `LinearSVC` (e.g., `maxIter`, `regParam`).
3.  **Try Other Models:** Experiment with `LogisticRegression`, `RandomForestClassifier`, `GBTClassifier`, etc.

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# --- This assumes 'pipeline' is defined and 'engineered_train_data' is ready ---

# paramGrid_lsvc = (
#     ParamGridBuilder()
#     .addGrid(lsvc.regParam, [0.1, 0.01, 0.001])
#     .addGrid(lsvc.maxIter, [10, 20, 30])
#     .build()
# )

# crossval = CrossValidator(
#     estimator=pipeline, # The entire pipeline is the estimator
#     estimatorParamMaps=paramGrid_lsvc,
#     evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"),
#     numFolds=3,  # Use 3-5 folds
#     parallelism=2, # Adjust based on your cluster resources
#     seed=42
# )

# print("Starting CrossValidator fitting...")
# # engineered_train_data.cache() # Cache data for CV
# cv_model = crossval.fit(engineered_train_data)
# # engineered_train_data.unpersist()
# print("CrossValidator fitting completed.")

# best_model_from_cv = cv_model.bestModel # This is a PipelineModel
# print("Best model obtained from CrossValidation.")

# # You can then evaluate this best_model_from_cv on engineered_validation_data
# # predictions_cv_validation = best_model_from_cv.transform(engineered_validation_data)
# # auc_cv_validation = evaluator_auc.evaluate(predictions_cv_validation)
# # print(f"AUC on validation data (with best CV model): {auc_cv_validation}")

# # Print best parameters
# # best_lsvc_model = best_model_from_cv.stages[-1] # Assuming LSVC is the last stage
# # print(f"Best regParam: {best_lsvc_model.getRegParam()}")
# # print(f"Best maxIter: {best_lsvc_model.getMaxIter()}")
```

## Phase 9: Prediction on Test Data and Submission

1.  **Prepare Test Data:**
    *   Apply the *exact same feature engineering steps* to `testHistory.csv` as you did for `trainHistory.csv`.
    *   The test data will not have a `repeater` or `label` column.
    *   Ensure the schema (column names and types) for features matches what the trained pipeline expects.

    ```python
    # --- Apply the SAME feature engineering to test_history_df ---
    # 1. Join test_history_df with offers_df
    # test_df_raw = test_history_df.join(offers_df, "offer")

    # 2. Apply all your feature engineering functions/logic used for train_df
    #    This will create columns like 'num_past_purchases_offer', 'avg_spend_customer', 'offer_category_engineered'
    #    For example:
    #    test_df_engineered = test_df_raw.withColumn("num_past_purchases_offer", lit(0.0)) # Placeholder
    #    test_df_engineered = test_df_engineered.withColumn("avg_spend_customer", lit(0.0)) # Placeholder
    #    test_df_engineered = test_df_engineered.withColumn("offer_category_engineered", lit("CAT_A")) # Placeholder

    # 3. Select the same feature columns as used in training, plus 'id'
    #    The column names must match 'categorical_cols_for_pipeline' and 'numerical_cols_for_pipeline'
    #    plus the 'id' column.
    # test_df_processed_for_prediction = test_df_engineered.select(
    #     ["id"] + categorical_cols_for_pipeline + numerical_cols_for_pipeline
    # )

    # print("Test data processed for prediction (schema):")
    # # test_df_processed_for_prediction.printSchema()
    ```

2.  **Transform:** Use your trained `model` (or `best_model_from_cv`) to make predictions.

    ```python
    # --- This assumes 'best_model_from_cv' (or 'model') is trained ---
    # --- and 'test_df_processed_for_prediction' is ready ---
    # print("Making predictions on processed test data...")
    # test_predictions = best_model_from_cv.transform(test_df_processed_for_prediction)
    # print("Test predictions made.")
    # # test_predictions.select("id", "rawPrediction", "prediction").show(5)
    ```

3.  **Format for Submission:**
    *   The submission usually requires `id` and the predicted probability (or a score related to it) of being a `repeater`.
    *   `LinearSVC`'s `rawPrediction` can be used. Check Kaggle's specific submission format.

    ```python
    # # Select id and the relevant prediction column
    # # For LinearSVC, rawPrediction is often used as the score.
    # submission_df = test_predictions.select(
    #     col("id"),
    #     col("rawPrediction").alias("probability") # Kaggle usually expects a probability or score
    # )
    #
    # print("Submission DataFrame sample:")
    # # submission_df.show(5)
    #
    # # Save to CSV
    # # submission_df.coalesce(1).write.csv(
    # #     "path/to/your_submission.csv", header=True, mode="overwrite"
    # # )
    # print("Submission file ready to be generated.")
    ```

## Summary of Key Points & Cautions

1.  **Feature Engineering is Paramount:** This is where you'll create the signals for your model. The provided code for this is a placeholder; you need to implement it thoroughly and consistently for train/test.
2.  **Correlation Timing:** Calculate correlations *after* feature engineering, on meaningful numerical features.
3.  **Pipeline Data Consistency:** Ensure column names and data types are consistent from feature engineering through to the pipeline stages.
4.  **Target Variable:** Use `label` (double type) for Spark ML.
5.  **Evaluation:** Use AUC on a validation set.
6.  **Iteration for "Best Solution":** The best model comes from iterating on features, tuning hyperparameters (e.g., with `CrossValidator`), and potentially trying different algorithms.
7.  **Consistent Processing:** Apply *identical* feature engineering and preprocessing to training and test data. Handle missing values and new categories consistently (e.g., `handleInvalid="keep"` or `"skip"` in `VectorAssembler`).
8.  **Data Leakage:** Be extremely careful that features for a given `(id, offerdate)` in `trainHistory` only use information available *before* that `offerdate`.
9.  **Spark Performance:** For large datasets, consider `cache()`/`persist()` for DataFrames that are used multiple times (like `engineered_train_data` during cross-validation) and `unpersist()` them when done. Adjust `parallelism` in `CrossValidator` based on your cluster.

This structured guide should help you build and refine your model. Good luck!
