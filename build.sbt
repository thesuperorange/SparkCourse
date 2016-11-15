name := "SparkCourse"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies ++= {
  val sparkV = "2.0.1"

  Seq(
    "org.apache.spark" %% "spark-core" % sparkV,
    "org.apache.spark" %% "spark-mllib" % sparkV,
    "org.apache.spark" %% "spark-sql" % sparkV,
    "org.apache.spark" %% "spark-graphx" % sparkV

  )
}