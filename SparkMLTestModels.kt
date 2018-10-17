package com.recruitics.ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.*


fun desicionTreeTest() {
  val session = SparkSession.builder().appName("MLTest").master("local[2]").orCreate

  // data w/ aprox. 6000 events
  var rawData = session.read().option("header", true).csv("resources/test4.csv")

  // cast columns to double
  rawData = rawData
      // average paid cpa for an event
      .withColumn("cpa", rawData.col("cpa").cast("Double"))
      .withColumn("age", rawData.col("age").cast("Double"))
      // paid visits
      .withColumn("visits", rawData.col("visits").cast("Double"))
      // paid applications
      .withColumn("applications", rawData.col("applications").cast("Double"))
      .withColumn("num_sources", rawData.col("num_sources").cast("Double"))
      // remove non-numeric symbols in onet code (and ignore anything after decimal)
      .withColumn("onetcode", regexp_replace(
          split(rawData.col("onetcode"), "\\.").getItem(0),
          "\\-", "")
          //.cast("Double")
      ).na().fill(99999L)


  // index location, city, and onet w/ lowest indices representing most popular labels
  // (note: these three features were found to have negligent effect on predicting paid applications)
  val locationIndexer = StringIndexer()
      .setInputCol("state")
      .setOutputCol("indexed_location")
      .setHandleInvalid("keep")

  val cityIndexer = StringIndexer()
      .setInputCol("city")
      .setOutputCol("indexed_city")
      .setHandleInvalid("keep")

  val onetIndexer = StringIndexer()
      .setInputCol("onetcode")
      .setOutputCol("indexed_onet")
      .setHandleInvalid("keep")


  // categorize sources where each event appears w/ hash function
  // (note: source categorization found to have negligible predictive ability)
  val tokenizer = Tokenizer()
      .setInputCol("source")
      .setOutputCol("sources")

  val hashingTF = CountVectorizer().setInputCol("sources").setOutputCol("sources_hash")

  // create feature vector (best performer: cpa and visits)
  val vectorAssembler = VectorAssembler()
      .setInputCols(arrayOf("cpa","visits"))
      .setOutputCol("features")

  // to be used by cross validator for testing different feature combos
  val vectorAssembler2 = VectorAssembler()
      .setInputCols(arrayOf("cpa"))
      .setOutputCol("features2")

  val vectorAssembler3 = VectorAssembler()
      .setInputCols(arrayOf("cpa","age","num_sources"))
      .setOutputCol("features3")

  val vectorAssembler4 = VectorAssembler()
      .setInputCols(arrayOf("cpa","age"))
      .setOutputCol("features4")

  val vectorAssembler5 = VectorAssembler()
      .setInputCols(arrayOf("cpa","indexed_location"))
      .setOutputCol("features5")

  val vectorAssembler6 = VectorAssembler()
      .setInputCols(arrayOf("cpa","indexed_onet"))
      .setOutputCol("features6")

  // clean the dataset
  val init = Pipeline()
      .setStages(arrayOf<PipelineStage>(
          locationIndexer,cityIndexer,onetIndexer,tokenizer, hashingTF,
          vectorAssembler, vectorAssembler2, vectorAssembler3,
          vectorAssembler4, vectorAssembler5, vectorAssembler6))

  var data = init.fit(rawData).transform(rawData)

  // used to specify size of applications label for random forest classifier
  val meta = NominalAttribute
      .defaultAttr()
      .withName("applications")
      .withNumValues(data.agg(max(data.col("applications"))).head().getDouble(0).toInt()+1)
      .toMetadata()
  data = data.withColumn("applications",data.col("applications").`as`("",meta))

  // create training and test data (70-30 split)
  val splits = data.randomSplit(doubleArrayOf(0.70, 0.3))
  val trainingData = splits[0]
  val testData = splits[1]

  // random foreset classifier w/ optimized hyperparameters
  val rf = RandomForestClassifier()
      .setLabelCol("applications")
      .setFeaturesCol("features")
      .setMaxBins(316)
      .setNumTrees(19)
      .setMaxMemoryInMB(4000)

  val pipeline = Pipeline()
      .setStages(arrayOf<PipelineStage>(rf))

  // to be used when optimizing hyper-parameters via cross-validation
  val paramGrid = ParamGridBuilder()
      //.addGrid(rf.numTrees(), intArrayOf(18,19,20))
      .addGrid(rf.featuresCol(), listOf("features","features2","features3","features4", "features5", "features6").asScalaIterable())
      .addGrid(rf.maxBins(), intArrayOf(316, 314, 318))
      .build()

  // Root mean squared error evaluator (can change metric to mean average error (mae) if desirable)
  val rmseEvaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

  // Evaluator w/ r^2 metric
  val r2Evaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("r2")

  // cross validator
  // set evaluator to either of the above
  // go to paramGrid val declaration to modify hyper-parameters to test
  val cv = CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(rmseEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(1)

  // comment out following line when not performing cross-validation
  // results of cross validation analysis can be found in the console log
  // by searching for "best set of parameters:"

  // val model = cv.fit(data)


  // comment out rest of method if performing cross validation
  // /*
  val model = pipeline.fit(trainingData)

  val predictions = model.transform(testData)

  //predictions = predictions.withColumn("predictedLabel",predictions.col("predictedLabel").cast("Double"))

  predictions.select( "applications","prediction","visits","cpa").show(800, false)

  val rmse = rmseEvaluator.evaluate(predictions)
  println("Root mean squared error = $rmse")

  val r2 = r2Evaluator.evaluate(predictions)
  println("R^2 = $r2")

  // pick whichever assertion seems most relevant
  //assert(rmse<4.0) { println("RMSE of $rmse greater than 4") }
  assert(r2>0.7) { println("R^2 of $r2 less than 0.7") }
  // */
}

fun generalizedLinearRegressionTest() {
  val session = SparkSession.builder().appName("MLTest").master("local[2]").orCreate

  // data w/ aprox. 6000 events
  var rawData = session.read().option("header", true).csv("resources/test4.csv")

  // cast to double
  rawData = rawData
      .withColumn("cpa", rawData.col("cpa").cast("Double"))
      .withColumn("age", rawData.col("age").cast("Double"))
      .withColumn("visits", rawData.col("visits").cast("Double"))
      .withColumn("num_sources", rawData.col("num_sources").cast("Double"))
      .withColumn("applications", rawData.col("applications").cast("Double"))
      .withColumn("onetcode", regexp_replace(
          split(rawData.col("onetcode"), "\\d\\d\\d\\d\\.").getItem(0),
          "\\-", "")
          .cast("Double"))
      .na().fill(99999L)

  // all with very small impact on results
  // indexed_location marginally improves current dataset's RMSE result,
  // however it is so marginal it is most likely due to chance
  val locationIndexer = StringIndexer()
      .setInputCol("state")
      .setOutputCol("indexed_location")
      .setHandleInvalid("keep")

  val cityIndexer = StringIndexer()
      .setInputCol("city")
      .setOutputCol("indexed_city")
      .setHandleInvalid("keep")

  val onetIndexer = StringIndexer()
      .setInputCol("onetcode")
      .setOutputCol("indexed_onet")
      .setHandleInvalid("keep")

  // create feature vector
  val vectorAssembler = VectorAssembler()
      .setInputCols(arrayOf("cpa","visits"))
      .setOutputCol("features")

  // to be used if performing cross validation analysis of different feature combos
  val vectorAssembler2 = VectorAssembler()
      .setInputCols(arrayOf("cpa","age"))
      .setOutputCol("features2")

  val vectorAssembler3 = VectorAssembler()
      .setInputCols(arrayOf("cpa","visits","indexed_city","indexed_location","age"))
      .setOutputCol("features3")

  val vectorAssembler4 = VectorAssembler()
      .setInputCols(arrayOf("cpa","visits","age","indexed_location"))
      .setOutputCol("features4")

  // create cleaned dataset
  val init = Pipeline()
      .setStages(arrayOf<PipelineStage>(
          locationIndexer,cityIndexer,onetIndexer,
          vectorAssembler, vectorAssembler2, vectorAssembler3,
          vectorAssembler4))
  val data = init.fit(rawData).transform(rawData)

  // create training and testing data (70-30 split)
  val splits = data.randomSplit(doubleArrayOf(0.7, 0.3))
  val trainingData = splits[0]
  val testData = splits[1]

  // regression evaluator w/ optimized hyper-parameters
  val regression = GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setLabelCol("applications")
      .setFeaturesCol("features")

  // r2 evaluator
  val r2Evaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("r2")
  // root mean squared error (rmse) evaluator
  val rmseEvaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

  // used by cross validator to test different ranges of hyper-parameters
  val paramGrid = ParamGridBuilder()
      .addGrid(regression.featuresCol(), listOf("features", "features2", "features3", "features4").asScalaIterable())
      //.addGrid(regression.family(), listOf("gaussian", "poisson", "gamma").asScalaIterable())
      //.addGrid(regression.link(), listOf("log", "identity").asScalaIterable())
      //.addGrid(regression.regParam(), doubleArrayOf(0.0,0.025,0.05,0.075,0.1,0.2,0.3))
      //.addGrid(regression.maxIter(), intArrayOf(5,10,1))
      .build()

  val pipeline = Pipeline()
      .setStages(arrayOf<PipelineStage>(regression))

  // set evaluator to either rmseEvaluator or r2Evaluator
  val cv = CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(rmseEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)
      .setParallelism(1)

  // comment out following line if not performing cross-validation analysis
  // results of cross validation analysis can be found in the console log
  // by searching for "best set of parameters:"

  // val cvModel = cv.fit(data)


  // comment out rest of method if performing cross-validation analysis
  // /*
  val model = regression.fit(trainingData)

  var predictions = model.transform(testData)

  predictions = predictions.withColumn("prediction",
      regexp_replace(round(predictions.col("prediction")),"-.*","0").cast("Double"))

  predictions.select("applications","prediction","features4").show(5000, false)

  val rmse = rmseEvaluator.evaluate(predictions)
  println("Root Mean Square Error = $rmse")
  val r2 = r2Evaluator.evaluate(predictions)
  println("r^2 = $r2")
  assert(r2>.75) { println("R^2 of $r2 less than than .75") }
  // assert(rmse>4) { println("rmse of $rmse greater than than 4") }
  // */
}

fun elasticSearchData() {

  val session = SparkSession.builder().appName("MLTest").master("local[2]").orCreate

  // working with 25000 events taken from ES database, with 30% in test and 70% in train
  var train = session.read().csv("resources/25000_feature_train.csv")
  train = train
      .withColumn("applications_paid", train.col("_c0").cast("Double"))
      .withColumn("spend", train.col("_c1").cast("Double"))
      .withColumn("visits_organic", train.col("_c2").cast("Double"))
      .withColumn("age", train.col("_c3").cast("Double"))

  var test = session.read().csv("resources/25000_feature_test.csv")
  test = test
      .withColumn("applications_paid", test.col("_c0").cast("Double"))
      .withColumn("spend", test.col("_c1").cast("Double"))
      .withColumn("visits_organic", test.col("_c2").cast("Double"))
      .withColumn("age", test.col("_c3").cast("Double"))

  // create feature vector
  val vectorAssembler = VectorAssembler()
      .setInputCols(arrayOf("visits_organic","age","spend"))
      .setOutputCol("features")

  // prepare datasets for training and testing
  train = vectorAssembler.transform(train)
  test = vectorAssembler.transform(test)

  // regression evaluator
  val regression = GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setLabelCol("applications_paid")
      .setFeaturesCol("features")

  // create model and test it on test dataset
  val model = regression.fit(train)
  var predictions = model.transform(test)

  // r2 evaluator
  val r2Evaluator = RegressionEvaluator()
      .setLabelCol("applications_paid")
      .setPredictionCol("prediction")
      .setMetricName("r2")
  // root mean squared error (rmse) evaluator
  val rmseEvaluator = RegressionEvaluator()
      .setLabelCol("applications_paid")
      .setPredictionCol("prediction")
      .setMetricName("mse")

  // get error metrics – can compare with SageMaker's Linear-Learner model (which tests error using MSE)
  val mse = rmseEvaluator.evaluate(predictions)
  println("Mean Square Error = $mse")
  val r2 = r2Evaluator.evaluate(predictions)
  println("r^2 = $r2")
}

/*
same as generalizedLinearRegressionTest, but testing on a larger set of data from jobs_aggregates
*/
fun generalizedLinearRegressionTest2() {
  val session = SparkSession.builder().appName("MLTest").master("local[2]").orCreate

  // data w/ aprox. 6000 events
  var rawData = session.read().option("header", true).csv("resources/job_aggregates.csv")

  // cast to double
  rawData = rawData
      .withColumn("spend", rawData.col("spend").cast("Double"))
      .withColumn("age", rawData.col("age").cast("Double"))
      .withColumn("visits", rawData.col("visitsorganic").cast("Double"))
      .withColumn("applications", rawData.col("applicationspaid").cast("Double"))
      .withColumn("onetcode", regexp_replace(
          split(rawData.col("onetcode"), "\\d.").getItem(0),
          "\\-", "")
          .cast("Double"))
      .na().fill(99999L)

  // all with very small impact on results
  // indexed_location marginally improves current dataset's RMSE result,
  // however it is so marginal it is most likely due to chance
  val locationIndexer = StringIndexer()
      .setInputCol("state")
      .setOutputCol("indexed_location")
      .setHandleInvalid("keep")

  val cityIndexer = StringIndexer()
      .setInputCol("city")
      .setOutputCol("indexed_city")
      .setHandleInvalid("keep")

  val onetIndexer = StringIndexer()
      .setInputCol("onetcode")
      .setOutputCol("indexed_onet")
      .setHandleInvalid("keep")

  // create feature vector
  val vectorAssembler = VectorAssembler()
      .setInputCols(arrayOf("spend","indexed_city","visits"))
      .setOutputCol("features")

  // to be used if performing cross validation analysis of different feature combos
  val vectorAssembler2 = VectorAssembler()
      .setInputCols(arrayOf("spend","visits"))
      .setOutputCol("features2")

  val vectorAssembler3 = VectorAssembler()
      .setInputCols(arrayOf("spend","visits","indexed_location","indexed_city"))
      .setOutputCol("features3")

  val vectorAssembler4 = VectorAssembler()
      .setInputCols(arrayOf("spend","visits","indexed_location"))
      .setOutputCol("features4")

  // create cleaned dataset
  val init = Pipeline()
      .setStages(arrayOf<PipelineStage>(
          locationIndexer,cityIndexer,onetIndexer,
          vectorAssembler, vectorAssembler2, vectorAssembler3,
          vectorAssembler4))
  var data = init.fit(rawData).transform(rawData)

  data = data.drop("city","state","onetcode")

  // create training and testing data (70-30 split)
  val splits = data.randomSplit(doubleArrayOf(0.7, 0.3))
  val trainingData = splits[0]
  val testData = splits[1]


  // regression evaluator w/ optimized hyper-parameters
  val regression = GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setLabelCol("applications")
      .setFeaturesCol("features3")

  // r2 evaluator
  val r2Evaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("r2")
  // root mean squared error (rmse) evaluator
  val rmseEvaluator = RegressionEvaluator()
      .setLabelCol("applications")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

  // used by cross validator to test different ranges of hyper-parameters
  val paramGrid = ParamGridBuilder()
      .addGrid(regression.featuresCol(), listOf("features", "features2", "features3", "features4").asScalaIterable())
      //.addGrid(regression.family(), listOf("gaussian", "poisson", "gamma").asScalaIterable())
      //.addGrid(regression.link(), listOf("log", "identity").asScalaIterable())
      .addGrid(regression.regParam(), doubleArrayOf(0.3,0.5,0.7,0.8,0.9))
      //.addGrid(regression.maxIter(), intArrayOf(5,10,1))
      .build()

  val pipeline = Pipeline()
      .setStages(arrayOf<PipelineStage>(regression))

  // set evaluator to either rmseEvaluator or r2Evaluator
  val cv = CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(rmseEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(1)

  // comment out following line if not performing cross-validation analysis
  // results of cross validation analysis can be found in the console log
  // by searching for "best set of parameters:"

  val model = cv.fit(data)


  // comment out the following line if performing cross-validation analysis
  // /*
  // val model = regression.fit(trainingData)

  var predictions = model.transform(testData)

  predictions = predictions.withColumn("prediction",
      regexp_replace(round(predictions.col("prediction")),"-.*","0").cast("Double"))

  predictions.select("applications","prediction","features3").show(5000, false)

  val rmse = rmseEvaluator.evaluate(predictions)
  println("Root Mean Square Error = $rmse")
  val r2 = r2Evaluator.evaluate(predictions)
  println("r^2 = $r2")
  assert(r2>.75) { println("R^2 of $r2 less than than .75") }
  // assert(rmse>4) { println("rmse of $rmse greater than than 4") }
  // */
}

// helper function to split a dataset into test and training csv files
// used to create data to be exported to sagemaker
// in future, function can be augmented to clean data (ex: index categorical data) before exporting to sagemaker
fun cleanAndExport() {
  val session = SparkSession.builder().appName("MLTest").master("local[2]").orCreate

  // data w/ aprox. 6000 events
  var rawData = session.read().option("header", true).csv("resources/job_aggregates2.csv")

  val locationIndexer = StringIndexer()
      .setInputCol("state")
      .setOutputCol("indexed_location")
      .setHandleInvalid("keep")

  val cityIndexer = StringIndexer()
      .setInputCol("city")
      .setOutputCol("indexed_city")
      .setHandleInvalid("keep")

  val onetIndexer = StringIndexer()
      .setInputCol("onetcode")
      .setOutputCol("indexed_onet")
      .setHandleInvalid("keep")
/*
  // create cleaned dataset
  val init = Pipeline()
      .setStages(arrayOf<PipelineStage>(
          *//*locationIndexer,*//*cityIndexer,onetIndexer))
  var data = init.fit(rawData).transform(rawData)

  data = data.drop("city","state","onetcode")*/
  rawData = rawData.drop("city","state","onetcode")

  // create training and testing data (70-30 split)
  val splits = rawData.randomSplit(doubleArrayOf(0.7, 0.3))
  val trainingData = splits[0]
  val testData = splits[1]

  val user = "faustinocortina"

  // set to local file location where you want test and training saved
  trainingData.write().format("com.databricks.spark.csv").save("/Users/$user/mldata/jobagg_train5.csv")
  testData.write().format("com.databricks.spark.csv").save("/Users/$user/mldata/jobagg_test5.csv")

}