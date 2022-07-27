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
import javax.management.modelmbean.ModelMBean;
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


public class Classfication {
    
    //Pythonモデル読み込み用クラス変数を定義。
    private SavedModelBundle model;


    private FileReader fileReader;
    private BufferedReader bf=null;

    private URI model_uri;
    private File input_file;
    private File model_file;

    private String model_path;
    private String input_file_path;
    private String line;

    private int index;

    private Float[][] data_f;
    private Float[][] input_f;
    private String[] data_s;

    private FloatNdArray input_matrix;
    private TFloat32 input_tens;
    private TFloat32 output_tens;



    public boolean isLoaded(){
        
        return model != null;

    }

    public void load() {
        
        if (isLoaded()){
            System.out.print("Model already has been loaded.");
        }else{

            try {

                loadModel();
                
            } catch (TensorFlowException e) {
                System.out.println("Model not found.");
                e.printStackTrace();
            } catch (URISyntaxException e) {
                System.out.println("URI is not corret.");
                e.printStackTrace();
            }
            
        }
    }
    

    public void unload(){
        model.close();
    }

    public void loadModel() throws TensorFlowException,URISyntaxException {

            
        //モデルの格納先をパス指定。
        model_uri = getClass().getResource("/my_mod").toURI();

        model_file = new File(model_uri);
        model_path = model_file.getPath();
        //モデルを読み込み。
        model = SavedModelBundle.load(model_path);

                
 }

        public void predict(){

        //(1, 28, 28)の三次元FLoat行列を定義。
        input_matrix=NdArrays.ofFloats(Shape.of(1,28,28));

        try {
            input_f=loaddata();
                    
             //input_matrix配列に値を挿入。
            for (int i=0; i<=27;i++){
            for(int j=0; j<=27; j++){
                
                 input_matrix.setFloat(input_f[i][j],0,i,j);
             }


            }
                        
        //input_matrix配列を引数にTensorオブジェクトを初期化。
            input_tens = TFloat32.tensorOf(input_matrix);
            output_tens= (TFloat32)model.session()
                                .runner()
                                .feed("serving_default_sequential_input",input_tens)
                                .fetch("StatefulPartitionedCall")
                                .run()
                                .get(0);
            System.out.println("");
            System.out.println("output: ");
            for (int i=0; i<10;i++){
                System.out.print(output_tens.getFloat(0,i)+", ");
            }
            System.out.println(" ");

        } catch (IOException e) {
            System.out.println("Input csv not found");
            e.printStackTrace();
           
        }

    }

        public Float[][] loaddata() throws IOException {
            input_file_path ="./src/main/java/input.csv";

           data_s = new String[28];
           data_f = new Float[28][28];
                
                input_file = new File(input_file_path);
                //入力データをCSVから読み込み。
                fileReader = new FileReader(input_file);
                bf = new BufferedReader(fileReader);
                index=0;

                while((line=bf.readLine()) != null) {
                    data_s = line.split(",");
                    for(int i=0;i<data_s.length;i++){
                        data_f[index][i] = (Float.parseFloat(data_s[i])/255.0f);
                    }
                    index++;
                }
              

            return data_f;
        }

        public void testSaveModelBundle(){
            System.out.println("Start...\n");
            load(); //モデル読み込み。
            predict();　//予測。
            unload(); //モデル解放。
            System.out.println("\nend...");
        }
    }
