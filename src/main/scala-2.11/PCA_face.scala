import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/17.
  */
object PCA_face {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local").appName("PCA").getOrCreate()
    val sc = spark.sparkContext
    val pcaRDD = sc.textFile("PCA_face.csv")


    val parsedData = pcaRDD.map { line =>
      org.apache.spark.ml.linalg.Vectors.dense(line.split(',').map(_.toDouble))
    }

    val df = spark.createDataFrame(parsedData.map(Tuple1.apply)).toDF("features")
    import org.apache.spark.ml.feature.PCA

    val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(36).fit(df)

    val pcaDF = pca.transform(df)

    val result = pcaDF.select("pcaFeatures")

  }
}
