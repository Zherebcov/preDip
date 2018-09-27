package zhe.lgtu.dip

import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import kotlin.reflect.KFunction0
import java.io.BufferedWriter
import java.io.FileWriter
import java.text.DecimalFormat


/**
 * Handwritten digits image classification on MNIST dataset (99% accuracy).
 * This example will download 15 Mb of data on the first run.
 * Supervised learning best modeled by CNN.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 */
object MyMnist {

    val log = LoggerFactory.getLogger(this::class.java)
    val basePath = System.getenv("SYMBOL")
    val CubData = ConfData(height = 12L, width = 10L, channels = 1L, batchSize = 108)
    val outputNum = 21
    val nEpochs = 30
    val seed = 1234


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val mnistDataSet = MnistDataSet(seed.toLong(), basePath, CubData, outputNum)
        val (trainIter, testIter) = mnistDataSet.getTestAndTrain()
        (0..1).forEach {  (12..12 step 1).forEach {
            first -> (listOf(25)).forEach{
            second -> (105..105 step 42).forEach {
            three ->
            val net = CreateNetwork(seed.toLong(), outputNum, CubData, "convInp10V2_7x7", listOf(first,second,three))
            net.net?.let { Fit(trainIter, testIter, net) }
        }} }}
    }


    @JvmStatic
    inline fun <T> NewTimed(text: String, build: KFunction0<T>): T {

        val start = System.currentTimeMillis()
        val set = build.invoke()
        val timeConsumedMillis = System.currentTimeMillis() - start
        log.info("$text {} ms", timeConsumedMillis)
        return set
    }

    @JvmStatic
    fun Fit(trainIter: MutableList<DataSet>, testIter: DataSet, net: CreateNetwork) {
        val acc = mutableListOf<Double>()
        val timeLog = Pair(mutableListOf<Long>(), mutableListOf<Long>())
        (0 until nEpochs).forEach {
            timeLog.first.add(NewTimed("Completed epoch ${it + 1}",false) { trainIter.forEach { net.net!!.fit(it) } })
            timeLog.second.add(NewTimed("Completed evaluation ${it + 1} ",false) {

                val eval = Evaluation(outputNum)
                val output = net.net!!.output(testIter.getFeatures())
                eval.eval(testIter.getLabels(), output)
                acc.add(eval.accuracy())
                //log.info(eval.accuracy().toString() /*+ eval.confusionMatrix()*/)
            })
        }
        val formatter = DecimalFormat("#0.00000")
        //val separator = "|"
        val textLog ="${net.name} | ${net.net?.numParams()} | ${net.nOut} | " +
                "${acc.map { formatter.format(it)}.joinToString()} | " +
                "${timeLog.first.joinToString()} | ${timeLog.second.joinToString()}"

        log.info("${net.name} | ${net.net?.numParams()} | ${net.nOut} | ${formatter.format(acc.last())} | " +
                "${formatter.format(timeLog.first.average()) } | ${formatter.format(timeLog.second.average()) }")

        val fw = FileWriter("result.txt", true)
        val bw = BufferedWriter(fw)
        bw.write(textLog)
        bw.newLine()
        bw.close()

       /* NewTimed("Save model") {
            ModelSerializer.writeModel(net.net!!, File("$basePath/minist-model.zip"), true)
        }*/
    }

    inline
    private fun NewTimed(text: String, printed: Boolean = true, function: () -> Unit): Long {
        val start = System.currentTimeMillis()
        function()
        val timeConsumedMillis = System.currentTimeMillis() - start
        if (printed) log.info("$text {} ms", timeConsumedMillis)
        return timeConsumedMillis
    }
}


data class ConfData(
        val height: Long = 10,
        val width: Long = 10,
        val channels: Long = 1, // single channel for grayscale images
        val batchSize: Int = 108)
