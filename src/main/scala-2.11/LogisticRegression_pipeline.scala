import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

var df = spark.read.option("header","true").option("inferSchema","true").csv("Log_reg_credit.csv")
val header= df.columns.filter(line=>line!=("RiskIndex"))
val indexer1 = new StringIndexer().setInputCol("Gender").setOutputCol("GenderIndex")
val indexer2 = new StringIndexer().setInputCol("Major").setOutputCol("MajorIndex")
val indexer3 = new StringIndexer().setInputCol("Risk").setOutputCol("label")
val header=Array("GenderIndex","MajorIndex","Age","GPA","HRS")
val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFeaturesCol("features")
val pipeline = new Pipeline().setStages(Array(indexer1,indexer2,indexer3,assembler, lr))
val model = pipeline.fit(df)
val lrModel = model.stages(4).asInstanceOf[LogisticRegressionModel]
