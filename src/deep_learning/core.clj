(ns deep-learning.core)

;; Neuron in a neural network
;; variables:
;; x = neuron inputs (x_1, x_2, ...)
;; w = weights for each input (w_1, w_2, ...) these will be learned
;; b = bias
;; f = f(z) = activation function, z = <x|w> + b
(defn neuron [x w b f]
  (f (reduce + (conj (map * x w) b))))


;; Sigmoid function f(z)=1/(1 + \exp{-z})
(defn sigmoid [z]
  (/ 1 (+ 1 (Math/exp (* -1 z)))))

;; derivative of sigmoid function
(defn d-sigmoid [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

;; plotting sigmoid function
(map sigmoid (range -5 5))
(neuron [1 2 3] [0.1 0.2 0.3] 1 sigmoid)

;; build the neural network up using Graph
