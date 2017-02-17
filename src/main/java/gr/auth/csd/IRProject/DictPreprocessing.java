package gr.auth.csd.IRProject;

import org.apache.commons.io.FileUtils;
import java.io.*;
import java.util.*;

/**
 * Preprocess raw files in order to alter TF weights according to dictionary.
 * Non-spark class.
 */
public class DictPreprocessing {

    static String dictionaryPath = "./src/main/resources/senticnet4.txt";
    static String inputRootDirectory = "./src/main/resources/data";

    /**
     * Execution entrypoint.
     * @param args CLI arguments
     */
    public static void main(String[] args) {
        // Read dictionary to main memory
        HashMap<String, Double> dictionary = readDictionary(dictionaryPath);
        Set<String> keys = dictionary.keySet();

        // Append relevant strong terms in each file
        File inputRootDirectoryFile = new File(inputRootDirectory);
        Collection<File> files = FileUtils.listFiles(inputRootDirectoryFile, new String[]{"txt"}, true);
        for (File file : files) {
            System.out.printf("File %s\n", file);
            appendToFile(file.toString(), dictionary, keys);
        }
    }

    /**
     * Method returning how many words to append according to word polarity.
     * @param weight The word's polarity
     * @return Number of words to append
     */
    public static int numberOfWordsToAppend(double weight) {
        if (weight < 0.65) {
            return 0;
        }
        else if (weight < 0.7) {
            return 2;
        }
        else if (weight < 0.8) {
            return 4;
        }
        else if (weight < 0.85) {
            return 8;
        }
        else if (weight < 0.9) {
            return 16;
        }
        else {
            return 32;
        }
    }

    /**
     * Append words to file.
     * @param path The path of the file to append to
     * @param dictionary The dictionary read into main memory
     * @param keys The key set of the dictionary
     * @return
     */
    public static void appendToFile(String path, HashMap<String, Double> dictionary, Set<String> keys) {
        System.out.printf("Appending to file: %s\n", path);
        BufferedReader br = null;
        Set<String> words = new HashSet<>();
        try {
            br = new BufferedReader(new FileReader(path));
            String currentLine;
            currentLine = br.readLine();
            String[] currentWords = new String[]{};
            while (currentLine != null) {
                currentWords = currentLine.split("(\\<.+\\>|\\W|[.,?:;!()])+");
                words = new HashSet<>(Arrays.asList(currentWords));
                currentLine = br.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br!=null) br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // Take intersection
        words.retainAll(keys);

        BufferedWriter bw = null;
        try {
            bw = new BufferedWriter(new FileWriter(path, true));
            for (String w : words) {
                int length = numberOfWordsToAppend(dictionary.get(w));
                for (int i=0;i<length;++i) {
                    bw.write(w);
                    bw.write(" ");
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw!=null) bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * Method reading dictionary into main memory.
     * @param path Path of the dictionary file
     * @return A hashmap with K: word, V: polarity
     */
    public static HashMap<String, Double> readDictionary(String path) {
        System.out.printf("Building dictionary");
        HashMap<String, Double> toBeReturned = new HashMap<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(path));
            String currentLine;
            currentLine = br.readLine();
            while (currentLine != null) {
                String[] tokens = currentLine.split(",");
                toBeReturned.put(tokens[0], Double.parseDouble(tokens[2]));
                currentLine = br.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br!=null) br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return toBeReturned;
    }
}
