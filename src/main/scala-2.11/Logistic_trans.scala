import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/16.
  */
object Logistic_trans {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("LinearRegression").getOrCreate()
    val sc = spark.sparkContext
    //correlation matrix
    val transRDD = sc.textFile("mllib_data/Transportation.csv")
    val transRDDNoHeader = transRDD.zipWithIndex().filter(_._2>0).map(line=>Vectors.dense(line._1.split(",").map({
      value=>
        if(value=="auto")0.0
          else if(value=="bus") 1.0
          else value.toDouble
    })))
    val correlMatrix: Matrix = Statistics.corr(transRDDNoHeader)
    correlMatrix.toArray.grouped(correlMatrix.numCols).toArray


    val trans = spark.read.option("header","true").option("inferSchema","true").csv("Transportation.csv")

    val header =trans.columns.filter(_!="Trans")

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val transformed = assembler.transform(trans)

    val labelIndexer= new StringIndexer() .setInputCol("Trans") .setOutputCol("label")
    val transformed2 = labelIndexer.fit(transformed).transform(transformed)

    val df2 = transformed2.select("features","label")

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFeaturesCol("features")

    val lrModel = lr.fit(df2)

    lrModel.coefficients


    import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary

    val trainingSummary = lrModel.summary

    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    //receiver-operating characteristic
    val roc = binarySummary.roc

    binarySummary.areaUnderROC

    binarySummary.predictions

    //precision & recall
    binarySummary.pr

    binarySummary.precisionByThreshold

    binarySummary.recallByThreshold

    binarySummary.fMeasureByThreshold


  //evaluator
    val prediction = lrModel.transform(df2)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction)
    println("Test Error = " + (1.0 - accuracy))
  }
}
