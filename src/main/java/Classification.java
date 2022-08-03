import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.TFloat32;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.IntNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.ShapeOps;
import org.tensorflow.exceptions.TensorFlowException;


class Classification {

    public static void main(String[] args) {


        Classification clf = new Classification();
        System.out.println("Start...\n");

        try {
            //モデル読み込み。
            SavedModelBundle model = clf.loadModel();
            //入力データ読み込み。
            Float[][] input_data = clf.loaddata();
            //入力データを学習データ用に整形。
            FloatNdArray tranning_data = clf.convert2Matrix(input_data);
            //予測を実行。
            TFloat32 prediction = clf.predict(model, tranning_data);
    
            System.out.println("");
            System.out.println("prediction: ");
            //予測データ出力。
            for (int i=0; i<10;i++){
                System.out.print(prediction.getFloat(0,i)+", ");
            }
            System.out.println(" ");
    
            System.out.println("\nend...");
            model.close();

            
        } catch (TensorFlowException e) {
            System.err.println("Model not found");
            e.printStackTrace();
        } catch(URISyntaxException e){
            System.err.println("Path to model is not correct");
            e.printSt   kTrace();
        } catch(IOException e){
            System.err.println("Input file not found");
            e.printStackTrace();
        }
    }


    public SavedModelBundle loadModel() throws TensorFlowException,URISyntaxException {

        //モデルの格納先をパス指定。
        URI model_uri = getClass().getResource("/my_mod").toURI();

        File model_file = new File(model_uri);
        String model_path = model_file.getPath();
        //モデルを読み込み。
        return SavedModelBundle.load(model_path);

 }

        public TFloat32 predict(SavedModelBundle model, FloatNdArray trainning_data){

        //input_matrix配列を引数にTensorオブジェクトを初期化。
            TFloat32 input_tens = TFloat32.tensorOf(trainning_data);
            TFloat32 output_tens= (TFloat32)model.session()
                                .runner()
                                .feed("serving_default_sequential_input",input_tens)
                                .fetch("StatefulPartitionedCall")
                                .run()
                                .get(0);
            return output_tens;
    }

      public FloatNdArray convert2Matrix(Float[][] input_data){
        //(1, 28, 28)の三次元FLoat行列を定義。
        FloatNdArray input_matrix=NdArrays.ofFloats(Shape.of(1,28,28));
        for (int i=0; i<=27;i++){
            for(int j=0; j<=27; j++){
                input_matrix.setFloat(input_data[i][j],0,i,j);
            }
        }

        return input_matrix;

      }
      
        public Float[][] loaddata() throws IOException {


           String input_file_path="./src/main/java/input.csv";

           String[] data_s = new String[28];
           Float[][] data_f = new Float[28][28];
                
           File input_file = new File(input_file_path);
           //入力データをCSVから読み込み。
           FileReader fileReader = new FileReader(input_file);
           BufferedReader bf = new BufferedReader(fileReader);
           int index=0;
           String line;

           while((line=bf.readLine()) != null) {
                data_s = line.split(",");
                for(int i=0;i<data_s.length;i++){
                    data_f[index][i] = (Float.parseFloat(data_s[i])/255.0f);
                }
                index++;
            }

          return data_f;
        }

}
    
