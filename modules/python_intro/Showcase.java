
/** A showcase of basic Python features, translated to Java. */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class Showcase {

    // ---- Variable Assignment ----

    static void variableAssignment() {
        // Python is dynamically typed, no type declarations needed.
        // In Java, types are declared explicitly and cannot change.
        String name = "Alice";
        int age = 21;
        double gpa = 3.85;
        boolean isEnrolled = true;

        System.out.println(((Object) age).getClass().getSimpleName()); // Integer
        System.out.println(((Object) gpa).getClass().getSimpleName()); // Double

        System.out.printf("%s is %d years old with a GPA of %.1f%n", name, age, gpa);

        // name = true; // Invalid in Java, a String cannot be reassigned to a boolean.
    }

    // ---- Collections ----

    static void collections() {
        // Lists
        List<Integer> scores = new ArrayList<>(List.of(88, 72, 95, 61, 84));

        System.out.println(scores.get(0)); // 88
        System.out.println(scores.get(scores.size() - 1)); // 84 (no negative indexing in Java)
        System.out.println(scores.subList(1, 4)); // [72, 95, 61]

        scores.add(90);
        System.out.println(scores); // [88, 72, 95, 61, 84, 90]

        // List comprehension to stream + filter + collect
        List<Integer> passing = scores.stream().filter(s -> s >= 70).collect(Collectors.toList());
        System.out.println(passing); // [88, 72, 95, 84, 90]

        // Tuples have no direct equivalent; a simple array or record works for
        // fixed-size data
        double[] point = { 1.5, 2.5 };
        System.out.println(point[0] + ", " + point[1]);

        // Dicts can be represented as a HashMap
        Map<String, Object> student = new HashMap<>();
        student.put("name", "Alice");
        student.put("age", 21);
        student.put("scores", scores);

        System.out.println(student.get("name")); // Alice
    }

    // ---- Control Flow ----

    static void controlFlow() {
        List<Integer> scores = new ArrayList<>(List.of(88, 72, 95, 61, 84, 90));

        // for-each loop
        // A regular indexed for loop has no Python counterpart
        for (int score : scores) {
            System.out.print(score + " ");
        }
        System.out.println();

        // while loop
        int i = 0;
        while (i < scores.size()) {
            System.out.print(scores.get(i) + " ");
            i++;
        }
        System.out.println();

        // Conditional
        if ("hello".equals("world")) {
            System.out.println("if");
        } else if ("foo".equals("foo")) {
            System.out.println("elif");
        } else {
            System.out.println("else");
        }
    }

    // ---- Functions ----

    static boolean isEven(int num) {
        return num % 2 == 0;
    }

    static void fireCallback(Consumer<Integer> callback) {
        callback.accept(3);
    }

    static void functions() {
        System.out.println(isEven(3)); // false

        // Lambda: Java lambdas match the signature of a functional interface
        // (Consumer<Integer> here)
        fireCallback(x -> System.out.println("Callback! ".repeat(x)));
    }

    // ---- Class ----
    // In Python, a class and the script that uses it can coexist in the same file.
    // In Java, the closest equivalent is a private static inner class.

    private static class Faz {
        int hars;

        public Faz(int hars) {
            this.hars = hars;
        }

        public void toreador() {
            System.out.println("har ".repeat(hars));
        }

        public static void party() {
            System.out.println("Let's eat!");
        }

        public static void outOfOrder() {
            throw new RuntimeException("Sorry");
        }

        @Override
        public String toString() {
            return "Stringified!";
        }
    }

    // ---- Main ----

    public static void main(String[] args) {
        variableAssignment();
        collections();
        controlFlow();
        functions();

        Faz f = new Faz(5);
        f.toreador();
        System.out.println(f);
    }
}
