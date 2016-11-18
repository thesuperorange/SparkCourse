/**
  * Created by user on 2016/11/13.
  */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object DecisionTree_wdbc {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("DecisionTree").getOrCreate()

    val wdbc= spark.read.option("inferSchema","true").csv("wdbc.data")

    val header = wdbc.columns.filter(x=>x!=("_c0")&& x!=("_c1"))


    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val labelIndexer= new StringIndexer() .setInputCol("_c1") .setOutputCol("label")

    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")

    val pipeline = new Pipeline() .setStages(Array(assembler,labelIndexer, dt))

    val model = pipeline.fit(wdbc)

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    val prediction = model.transform(wdbc)
    prediction.select("prediction", "label", "features").show(5)


    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(prediction)
    println("Test Error = " + (1.0 - accuracy))

  }

}
