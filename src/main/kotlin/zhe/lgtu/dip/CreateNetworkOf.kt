package zhe.lgtu.dip

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import zhe.lgtu.dip.MyMnist.CubData
import java.util.HashMap


class CreateNetwork(val seed: Long, val  outputNum: Int, val CubData: ConfData,val name:String, val nOut: List<Int>) {

    val log = LoggerFactory.getLogger(this::class.java)
    var net:MultiLayerNetwork? = null


    fun convInp10_2L_3x3(nOut: List<Int>){

        val convLay = Pair(3, 3)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
               // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(nOut[1])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10V2_2L_5x5(nOut: List<Int>){

        val convLay = Pair(5, 5)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10_6x6(nOut: List<Int>){

        val convLay = Pair(6, 6)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10_7x7(nOut: List<Int>){

        val convLay = Pair(7, 7)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10V2_7x7(nOut: List<Int>){

        val convLay = Pair(7, 7)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10_9x9(nOut: List<Int>){

        val convLay = Pair(9, 9)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(3, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp10V3_2L_5x5(nOut: List<Int>){

        val convLay = Pair(5, 5)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder(3, 3)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[1])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }

    fun convInp12_2L_3x3(nOut: List<Int>){

        val convLay = Pair(4, 4)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.0025

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                // .l2(0.00025)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(CubData.channels)
                        .stride(1, 1)
                        .nOut(nOut[0])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(nOut[1])
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(nOut[2]).build())
                .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CubData.height, CubData.width, CubData.channels))
                .backprop(true).pretrain(false).build()

        this.net = MultiLayerNetwork(conf)
        net!!.init()
        //net.setListeners(ScoreIterationListener(20000 / batchSize / 20))
        //log.debug("Total num of params: {}", net!!.numParams())
    }
    init {
        log.debug("Network configuration and training...")
        when (name){
            "convInp10_2L_3x3" -> convInp10_2L_3x3(nOut)
            "convInp12_2L_3x3" -> convInp12_2L_3x3(nOut)
            "convInp10V2_2L_5x5" -> convInp10V2_2L_5x5(nOut)
            "convInp10_6x6" -> convInp10_6x6(nOut)
            "convInp10_7x7" -> convInp10_7x7(nOut)
            "convInp10_9x9" -> convInp10_9x9(nOut)
            "convInp10V2_7x7" -> convInp10V2_7x7(nOut)
            else -> IllegalArgumentException("$name is wrong argument to name")
        }
    }
}
