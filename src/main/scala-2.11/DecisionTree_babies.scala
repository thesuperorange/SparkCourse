import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/17.
  */
object DecisionTree_babies {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("DecisionTree").getOrCreate()

    val baby = spark.read.option("header","true").option("inferSchema","true").option("nullValue","NA").option("delimiter"," ").csv("babies.txt")
    val baby2 = baby.na.drop

    val header = baby.columns.filter(_!=("bwt"))

    import org.apache.spark.ml.feature.VectorAssembler
    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    var transformed = assembler.transform(baby2)

    transformed =transformed.withColumn("label",transformed("bwt")).select("label","features")


    import org.apache.spark.ml.regression.DecisionTreeRegressor
    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")
    val dtModel = dt.fit(transformed)

    dtModel.toDebugString
    dtModel.depth

    val predictions = dtModel.transform(transformed)

    predictions.show
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    val evaluator = new RegressionEvaluator()  .setLabelCol("label")  .setPredictionCol("prediction") .setMetricName("rmse")

    val RMSE= evaluator.evaluate(predictions)



  }
}
