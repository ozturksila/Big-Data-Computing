import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.sources.In;
import org.spark_project.guava.collect.Iterables;
import org.spark_project.guava.collect.Iterators;
import scala.Tuple2;
import java.io.IOException;
import java.util.*;

public class G011HW1 {
    public static void main (String[] args) throws IOException {

        SparkConf conf      = new SparkConf (true).setAppName ("HW1");
        JavaSparkContext sc = new JavaSparkContext (conf);
        sc.setLogLevel ("WARN");

        int K    = Integer.parseInt (args[0]); // Read number of partitions
        int H    = Integer.parseInt (args[1]); // Read number of best products to display
        String S = new String       (args[2]); // Read Country


        // Step 1 : Read input file and subdivide it into K random partitions
        JavaRDD<String> RawData = sc.textFile (args[3]).repartition (K).cache ();
        System.out.println ("Number of total transactions = " + RawData.count ());


        // Step 2 : productCustomer DISTINCT pairs (P,C)
        JavaPairRDD<String, Integer> product_customer;

        product_customer = RawData
                .flatMapToPair ((document) -> {    // <-- MAP PHASE (R1)
                    // Split each line of transaction
                    String[] tokens = document.split ( ",");

                    HashMap<String, Integer> distinct_pairs  = new HashMap<> ();
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<> ();
                    // Set<Object> distinct_pairs = new HashSet<>();
                    int qty = Integer.parseInt (tokens[3]);

                    if (qty > 0 && (S.equals ("all") || tokens[7].equals (S)))
                        // FIXME: DISTINCT if condition didn't work
                        //if (distinct_pairs.get (tokens[1]) != Long.parseLong (tokens[6]))
                        pairs.add(new Tuple2<>(tokens[1], Integer.parseInt(tokens[6]))); //set if NOT exists condition

                    //for (Map.Entry<String, Long> e : distinct_pairs.entrySet ()) {
                    // FIXME: DISTINCT if condition could be introduced here as well
                    //  pairs.add (new Tuple2<>(e.getKey (), e.getValue ())); //set if NOT exists condition
                    //}
                    return pairs.iterator ();
                });
        JavaPairRDD<String, Integer> distinct_pairs = product_customer.groupByKey()
                .flatMapToPair((element )-> {
                    //HashMap<String, Integer> hashMap  = new HashMap<> ();
                    ArrayList<Tuple2<String, Integer>> pairs = new ArrayList<> ();

                    Set<Integer> hash_Set = new HashSet<Integer>();

                    for (Integer t : element._2()) {
                        hash_Set.add(t);
                    }
                    for (Integer t : hash_Set) {
                        pairs.add(new Tuple2<>(element._1(), t));
                    }

                    return pairs.iterator();
                });


        System.out.println ("Number of product customer pairs (NOT distinct) = " + distinct_pairs.count());
        //Step3
        JavaPairRDD<String, Integer> product_popularity1;

        product_popularity1 = distinct_pairs.groupByKey().mapToPair((element)-> {

            return new Tuple2<>(element._1(), Iterables.size(element._2()));
        } );

        // Step 4
        JavaPairRDD<String, Integer> product_popularity2;

        product_popularity2 = product_popularity1.reduceByKey(
                (v1, v2) -> Math.max(v1, v2)
        );


        //Step 5
        if ( H> 0) {
            // Step 4 : we swap keys and values, then we use sortByKeys, and we swap again.
            JavaPairRDD<Integer, String> swap = product_popularity1.mapToPair(x -> x.swap());
            JavaPairRDD<Integer, String> sorted = swap.sortByKey(false);
            JavaPairRDD<String, Integer> ordered = sorted.mapToPair(x -> x.swap());

            //Step 6
            for (int i = 0; i < H; i++) {
                System.out.println("Product: " + ordered.keys().take(10).get(i).toString() + " count:  " + ordered.values().take(10).get(i).toString());
            }
        }
        //Step 6
        if (H==0)
        {
            JavaPairRDD<String, Integer> sorted = product_popularity1.sortByKey();
            for (int i = 0; i < product_popularity1.count(); i++)
            {
                System.out.println ("Product: " + product_popularity1.keys ().take (10).get (i).toString () + " count:  " + product_popularity1.values ().take (10).get (i).toString ());
                System.out.println ("Product: " + product_popularity2.keys ().take (10).get (i).toString () + " count:  " + product_popularity2.values ().take (10).get (i).toString ());
            }
        }
    }

}