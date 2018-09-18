package zhe.lgtu.dip

import java.io.File

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import kotlin.reflect.KFunction0

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
    val height = 10
    val width = 10
    val channels = 1 // single channel for grayscale images
    val outputNum = 21 // 10 digits classification
    val batchSize = 108
    val nEpochs = 30
    val seed = 1234

    val timeLearn = listOf<Int>()
    val timeEval = listOf<Int>()


    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        val mnistDataSet = MnistDataSet(seed.toLong(), basePath, height.toLong(), width.toLong(), channels.toLong(), batchSize, outputNum)
        val (trainIter, testIter) = mnistDataSet.getTestAndTrain()
        (3..10 step 2).forEach {
            first -> (20..60 step 10).forEach{
            second -> (50..500 step 50).forEach {
            free ->
            val net = CreateNetwork(seed.toLong(), channels.toLong(), outputNum, height.toLong(), width.toLong(), "convInp10_2L_3x3", listOf(first,second,free))
            net.net?.let { Fit(trainIter, testIter, net) }
        }} }
    }

    @JvmStatic
    inline fun <T> NewTimed(text: String, build: KFunction0<T>): T {

        var start = System.currentTimeMillis()
        val set = build.invoke()
        var timeConsumedMillis = System.currentTimeMillis() - start
        log.info("$text {} ms", timeConsumedMillis)
        return set
    }

    @JvmStatic
    fun Fit(trainIter: MutableList<DataSet>, testIter: DataSet, net: CreateNetwork) {
        var acc = mutableListOf<Double>()
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
        log.info("For model ${net.name} params ${net.net?.numParams()} with parameters ${net.nOut} acc - ${acc.last()} " +
                "timeTrain - ${timeLog.first.last()} timeEval - ${timeLog.second.last()}")

       /* NewTimed("Save model") {
            ModelSerializer.writeModel(net.net!!, File("$basePath/minist-model.zip"), true)
        }*/
    }

    inline
    private fun NewTimed(text: String, printed: Boolean = true, function: () -> Unit): Long {
        var start = System.currentTimeMillis()
        function()
        var timeConsumedMillis = System.currentTimeMillis() - start
        if (printed) log.info("$text {} ms", timeConsumedMillis)
        return timeConsumedMillis
    }
}
