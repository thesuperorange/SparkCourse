import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, DataType}
/**
  * Created by user on 2016/11/16.
  */
object LinearRegression_dogs {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("LinearRegression").getOrCreate()
    val sc = spark.sparkContext
    val dogs = sc.textFile("dogs2.txt")
    dogs.map(_.split(" +").mkString(",")).repartition(1).saveAsTextFile("dogs")


    val dogs2 = spark.read.option("inferSchema","true").option("header","true").csv("dogs/part*")

    val header =dogs2.columns.filter(idx=>idx!="city" && idx!="adoptedR")

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")
    val transformed = assembler.transform(dogs2)
    val df2 = transformed.withColumn("label",transformed("adoptedR").cast(DoubleType)).select("label","features")


    import org.apache.spark.ml.regression.LinearRegression
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) .setLabelCol("label").setFeaturesCol("features")

    val model = lr.fit(df2)

    model.intercept
    model.coefficients

    model.summary.devianceResiduals
    model.summary.residuals
    model.summary.predictions
    val predictions = model.transform(df2)

    predictions.show
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    val evaluator = new RegressionEvaluator()  .setLabelCol("label")  .setPredictionCol("prediction") .setMetricName("rmse")

    val RMSE= evaluator.evaluate(predictions)







  }
}
