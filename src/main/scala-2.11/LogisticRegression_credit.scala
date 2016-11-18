import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/16.
  */
object LogisticRegression_credit {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("LogisticRegression").getOrCreate()
    var df = spark.read.option("header","true").option("inferSchema","true").csv("Log_reg_credit.csv")
    df = df.drop("OBS")
    df.select("Major").distinct().show

    df.select("Gender").distinct().show

    df.select("Risk").distinct().show
    import org.apache.spark.ml.feature.StringIndexer

    val indexer1 = new StringIndexer().setInputCol("Gender").setOutputCol("GenderIndex")
    val indexer2 = new StringIndexer().setInputCol("Major").setOutputCol("MajorIndex")
    val indexer3 = new StringIndexer().setInputCol("Risk").setOutputCol("RiskIndex")

    df= indexer1.fit(df).transform(df).drop("Gender")
    df = indexer2.fit(df).transform(df).drop("Major")
    df= indexer3.fit(df).transform(df).drop("Risk")
    import org.apache.spark.ml.feature.VectorAssembler

    val header= df.columns.filter(line=>line!=("RiskIndex"))

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val transformed = assembler.transform(df)

    val df2 = transformed.withColumn("label",transformed("RiskIndex")).select("features","label")
    df2.write.parquet("credit")
//------

    import org.apache.spark.ml.classification.LogisticRegression

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFeaturesCol("features")

    val lrModel = lr.fit(df2)
    lrModel.coefficients
    lrModel.intercept

//-----------------
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

  }
}
