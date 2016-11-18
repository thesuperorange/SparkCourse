/**
  * Created by user on 2016/11/15.
  */

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

object StatisticSummary {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("Correlation").getOrCreate()
    val sc = spark.sparkContext

    val carRDD = sc.textFile("mllib_data/car.csv")

    val carNoHeader = carRDD.zipWithIndex().filter(_._2>0).map(line=>Vectors.dense(line._1.split(",").map(_.toDouble)))


    import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

    val summary: MultivariateStatisticalSummary = Statistics.colStats(carNoHeader)

    println(summary.mean)
    println(summary.variance)
    println(summary.numNonzeros)

    import org.apache.spark.mllib.linalg._

    val correlMatrix: Matrix = Statistics.corr(carNoHeader)
    correlMatrix.toArray.grouped(correlMatrix.numCols).toArray


  }
}
