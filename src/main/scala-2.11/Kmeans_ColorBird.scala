import org.apache.spark.sql.SparkSession

/**
  * Created by user on 2016/11/17.
  */
object Kmeans_ColorBird {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().master("local").appName("Kmeans").getOrCreate()
    val df = spark.read.option("inferSchema","true").option("header", "true").csv("ColorBird.csv").drop("_c0")

    import org.apache.spark.ml.feature.VectorAssembler

    val header = df.columns.filter(line=>line!=("x")&& line!=("y"))

    val assembler = new VectorAssembler().setInputCols(header).setOutputCol("features")
    val transformed = assembler.transform(df)

    val df2 = transformed.select("features","x","y")

    import org.apache.spark.ml.clustering.KMeans
    val kmeans = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")

    val model = kmeans.fit(df2)
    //看中心點
    model.clusterCenters.foreach(println)

    val result = model.transform(df2)
    //先註冊table 
    result.createOrReplaceTempView("kmeansTable")
    spark.sql("SELECT * from kmeansTable where prediction=1").count


  }
}
