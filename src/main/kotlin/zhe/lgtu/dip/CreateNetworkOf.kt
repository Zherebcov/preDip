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
import java.util.HashMap


class CreateNetwork(val seed: Long, val channels: Long,val  outputNum: Int,val  height: Long,val  width: Long,val name:String, val nOut: List<Int>) {

    val log = LoggerFactory.getLogger(this::class.java)
    var net:MultiLayerNetwork? = null


    fun convInp10_2L_3x3(nOut: List<Int>){

        val convLay = Pair(3, 3)
        val lrSchedule = HashMap<Int, Double>()
        lrSchedule[0] = 0.01
        lrSchedule[600] = 0.006
        lrSchedule[1000] = 0.003

        val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .updater(Nesterovs(MapSchedule(ScheduleType.ITERATION, lrSchedule)))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, ConvolutionLayer.Builder(convLay.first, convLay.second)
                        .nIn(channels)
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
                .setInputType(InputType.convolutionalFlat(height, width, channels))
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
            else -> IllegalArgumentException("$name is wrong argument to name")
        }
    }
}
