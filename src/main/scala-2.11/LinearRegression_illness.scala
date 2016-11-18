import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
/**
  * Created by user on 2016/11/16.
  */
object LinearRegression_illness {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("LinearRegression").getOrCreate()
    val sc = spark.sparkContext
    //correlation matrix
  val illRDD = sc.textFile("illness.csv")
    val illRDDNoHeader = illRDD.zipWithIndex().filter(_._2>0).map(line=>Vectors.dense(line._1.split(",").map(_.toDouble)))
    val correlMatrix: Matrix = Statistics.corr(illRDDNoHeader)
    correlMatrix.toArray.grouped(correlMatrix.numCols).toArray

    //--------------

    val df = spark.read.option("header", "true") .option("inferSchema", "true") .csv("illness.csv")
    import org.apache.spark.mllib.linalg.{Vectors,Vector}
    import org.apache.spark.sql.types.DoubleType

    //user defined function
    val toVec3    = udf[ org.apache.spark.ml.linalg.Vector, Double, Double,  Double] { (r,g,b) =>
      org.apache.spark.ml.linalg.Vectors.dense(r,g,b)
    }
    //satisfy-> label,  age anxiety illness-> features
    val df2 = df.withColumn(
      "features",
      toVec3(
        df("age"),
        df("illness"),
        df("anxiety")
      )
    ).withColumn("label", df("satisfy").cast(DoubleType)).select("label","features")


    import org.apache.spark.ml.regression.LinearRegression
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) .setLabelCol("label").setFeaturesCol("features")

    val model = lr.fit(df2)
    model.coefficients
    model.intercept
    model.summary.devianceResiduals
    model.summary.residuals
    model.summary.predictions




  }
}
