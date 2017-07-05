<?php

namespace Perceptron;

/**
 * Simple Perceptron
 */
class Perceptron {
    /**
     * @var array
     */
	protected $vectorSize;

    /**
     * @var float
     */
	protected $bias;

    /**
     * @var float
     */
	protected $learningRate;

    /**
     * @var array
     */
	protected $weights;

    /**
     * @var int
     */
	protected $iterations;

    /**
     * @var float
     */
	protected $errorSum = 0;

    /**
     * @var float
     */
	protected $iterationError = 0;

	/*
	 * Constructor method
	 * Initialize weights
	 *
	 * @param int $vectorSize
	 * @param float $bias
	 * @param float $learningRate
	 */
	function __construct(int $vectorSize, float $bias = 1, float $learningRate = 0.5) {
		if ($vectorSize < 1) {
            throw new \InvalidArgumentException();
        } elseif ($learningRate <= 0 || $learningRate > 1) {
            throw new \InvalidArgumentException();
        }

		$this->vecLength = $vectorSize;
        $this->bias = $bias;
        $this->learningRate = $learningRate;

        for ($i = 0; $i < $vectorSize; $i++) { 
        	$this->weights[$i] = $this->random();
        }
	}

	/*
     * Predict method takes an input vector and returns a guess
     *
     * @param array $inputs
     *
     * @return int result of guess
     * @throws \InvalidArgumentException
     */
    public function predict(array $inputs) : int {
        if (!is_array($inputs) || count($inputs) != $this->vecLength) {
            throw new \InvalidArgumentException();
        }
        $result = $this->compute($inputs, $this->weights) + $this->bias;
        $this->output = $this->activation($result);
        return $this->output;
    }

    /*
     * Training method takes an input vector and the correct prediction
     * - tests the inputs with current weights
     * - alters each weight based on the output
     * - alters bias
     * - alters error sum and iteration error
     *
     * @param array $inputs
     * @param int $predicted
     *
     * @throws \InvalidArgumentException
     */
    public function train(array $inputs, float $predicted) {
        if (!is_array($inputs) || !($predicted == -1 || $predicted == 1)) {
            throw new \InvalidArgumentException();
        }
        // Increment iteration
        $this->iterations += 1;
        // Get output of test
        $output = $this->predict($inputs);
        // Loop through weight vector
        for ($i = 0; $i < $this->vecLength; $i++) {
        	// Alter each weight
            $this->weights[$i] = $this->alterWeight($this->weights[$i], $predicted, $output, $inputs[$i]);
        }
        // Alter the bias
        // predicted value - test output
        $this->bias += ((int) $predicted - (int) $output);
        // Alter error sum
        $this->errorSum += (int) $predicted - (int) $output;
        // Set iteration error
        $this->iterationError = 1 / $this->iterations * $this->errorSum;
    }

    /*
     * Save the trained data
     *
     * @return array
     */
    public function saveState() : array {
        return [
            'bias' => $this->bias,
            'learningRate' => $this->learningRate,
            'vecLength' => $this->vecLength,
            'weights' => $this->weights
        ];
    }

    /*
     * Load a trained state
     *
     * @param array $state
     */
    public function loadState(array $state) {
        if(empty($state['bias']) || !is_numeric($state['bias'])) {
            throw new \InvalidArgumentException();
        }
        if(empty($state['learningRate']) || !is_numeric($state['learningRate']) || $state['vecLength'] <= 0  || $state['vecLength'] > 0) {
            throw new \InvalidArgumentException();
        }
        if(empty($state['vecLength']) || !is_numeric($state['vecLength']) || $state['vecLength'] > 0) {
            throw new \InvalidArgumentException();
        }
        if(empty($state['weights']) || !is_array($state['weights'])) {
            throw new \InvalidArgumentException();
        }
        $this->bias = $state['bias'];
        $this->learningRate = $state['learningRate'];
        $this->vecLength = $state['vecLength'];
        $this->weights = $state['weights'];
    }


    /*
	 * Get the weights
	 *
	 * @return array
     */
    public function getWeights() : array {
    	return $this->weights;
    }

    /*
     * Set the weights
     *
     * @param array $weight
     */
    public function setWeights(int $weights) {
        if (!is_array($weights)) {
            throw new \InvalidArgumentException();
        }
        $this->weights = $weights;
    }

    /*
     * Get the bias
     *
     * @return int
     */
    public function getBias() : int {
        return $this->bias;
    }

    /*
     * Set the bias
     *
     * @param int $bias
     */
    public function setBias(int $bias) {
        if (!is_numeric($bias)) {
            throw new \InvalidArgumentException();
        }
        $this->bias = $bias;
    }

    /*
     * Get the learning rate
     *
     * @return int
     */
    public function getLearningRate() : int {
        return $this->learningRate;
    }

    /*
     * Set the learning rate
     *
     * @param int $rate
     */
    public function setLearningRate(int $rate) {
        if (!is_numeric($rate) || $learningRate <= 0 || $learningRate > 1) {
            throw new \InvalidArgumentException();
        }
        $this->learningRate = $rate;
    }

    /*
     * Do the calculation
     * Add together the products of each input and their weight
     *
     * @param array $vec1 inputs vector
     * @param array $vec2 weighs vector
     *
     * @return float
     */
    protected function compute(array $inputs, array $weights) : float {
        return array_sum(
            array_map(
                function ($i, $w) {
                    return $i * $w;
                },
                $inputs,
                $weights
            )
        );
    }

    /*
	 * Alter the weight
	 *
	 * @param int $weight
	 * @param int $predicted
	 * @param int $output
	 * @param int $inputs
	 *
	 * @return int
     */
    protected function alterWeight(int $weight, int $predicted, int $output, int $inputs) : int {
    	return $weight + $this->learningRate * ((int)$predicted - (int)$output) * $inputs;
    }

	/*
	 * Returns random float between -1 and 1 for initializing weights
	 *
	 * @return float
	 */
	protected function random() : float {
		return rand()/getrandmax() * 2 - 1;
	}

	/*
     * The sign activation gunction
     *
     * @param float $input
     *
     * @return int
     */
	protected function activation(float $input) : float {
		if($input >= 0) {
			return 1;
		}
		return -1;
	}

}