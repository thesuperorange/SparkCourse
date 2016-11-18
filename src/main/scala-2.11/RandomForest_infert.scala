import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/13.
  */

import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
object RandomForest_infert {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("RandomForest").getOrCreate()
    val infert= spark.read.option("header","true").option("inferSchema","true").csv("infert.csv")


    val header = Array("age","parity","induced","spontaneous")
    //轉換double
/*
    for (castme <- header) {
      infert= infert.withColumn(castme, infert(castme).cast(DoubleType))
    }*/


    val labelIndexer= new StringIndexer() .setInputCol("case") .setOutputCol("label")

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val Array(trainingData, testData) = infert.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(10)


    val pipeline = new Pipeline().setStages(Array(labelIndexer,assembler,  rf))

    val model = pipeline.fit(trainingData)

    val treeModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    treeModel.toDebugString

    val prediction = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction)
    println("Test Error = " + (1.0 - accuracy))


  }
}
