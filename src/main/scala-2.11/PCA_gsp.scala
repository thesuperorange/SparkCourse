import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.PCA
/**
  * Created by user on 2016/11/14.
  */
object PCA_gsp {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("PCA").getOrCreate()
    val gsp = spark.read.option("inferSchema","true").option("header","true").csv("pca_gsp.csv")


    val header = gsp.columns.filter(_!="State")


    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")

    val transformed = assembler.transform(gsp)


    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(10)
    val pcaModel = pca.fit(transformed)
    pcaModel.pc
    pcaModel.explainedVariance


    val pcaDF = pcaModel.transform(gsp)
    val result = pcaDF.select("pcaFeatures")
    result.show

  }
}
