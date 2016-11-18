import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/17.
  */
object CrossValidation_trans {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("CrossValidation").getOrCreate()
    val sc = spark.sparkContext
    //correlation matrix
    val transRDD = sc.textFile("Transportation.csv")
    val transRDDNoHeader = transRDD.zipWithIndex().filter(_._2>0).map(line=>Vectors.dense(line._1.split(",").map({
      value=>
        if(value=="auto")0.0
        else if(value=="bus") 1.0
        else value.toDouble
    })))
    val correlMatrix: Matrix = Statistics.corr(transRDDNoHeader)
    correlMatrix.toArray.grouped(correlMatrix.numCols).toArray

//-----------------
    val trans = spark.read.option("header","true").option("inferSchema","true").csv("Transportation.csv")

    val header =trans.columns.filter(_!="Trans")

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val labelIndexer= new StringIndexer() .setInputCol("Trans") .setOutputCol("label")


    val lr = new LogisticRegression().setMaxIter(10)
    val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, lr))

    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.01)).addGrid(lr.elasticNetParam, Array(0.01,0.1, 0.8))  .build()

    val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)  // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(trans)

    val model = cvModel.bestModel

    cvModel.avgMetrics

    val prediction = cvModel.transform(trans)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction)
    println("Test Error = " + (1.0 - accuracy))
  }
}
