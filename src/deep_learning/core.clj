(ns deep-learning.core
  (:use plumbing.core)
  (:require [plumbing.graph :as graph]
            [schema.core :as s]
            [clojure.core.reducers :as r]))

;;; Schema definitions for type-checking and documentation
(def DoubleVec [double])

;;; Neural Network functions

(s/defrecord)
(s/defn neuron
  "Neuron in a neural network.
  variables:
  x = neuron inputs (x_1, x_2, ...)
  w = weights for each input (w_1, w_2, ...) these will be learned
  b = bias
  f = f(z) = activation function, z = <x|w> + b"
  [x :- DoubleVec
   w :- DoubleVec
   b :- s/Number f]
  {:pre (= (count x) (count w))}
  (f (r/reduce + (conj (map * x w) b))))

(s/defn sigmoid
  "Sigmoid function f(z)=1/(1 + \exp{-z})"
  [z :- s/Number]
  (/ 1 (+ 1 (Math/exp (* -1. z)))))

(s/defn d-sigmoid
  "derivative of sigmoid function"
  [z :- s/Number]
  (* (sigmoid z) (- 1. (sigmoid z))))

(defn neural-layer [elements])

;; plotting sigmoid function
(map sigmoid (range -5 5))
(neuron [1 2 3] [0.1 0.2 0.3] 1 sigmoid)

;; build the neural network up using Graph
(def neural-net-graph
  "A graph describing the structure of the neural network"
  { :hidden-layer (fnk [x w] [(neuron x (w 0) 1 sigmoid) (neuron x (w 1) 1 sigmoid) (neuron x (w 2) 1 sigmoid)] )
    :output-unit (fnk [hidden-layer w-output-layer] (neuron hidden-layer w-output-layer 1 sigmoid))})

(def neural-net (graph/eager-compile neural-net-graph))

(neural-net {:x [1 2 3] :w [[0.1 0.2 -0.1] [0.1 0.2 0.1] [0.5 -0.1 -0.2]] :w-output-layer [1 1 1]})
