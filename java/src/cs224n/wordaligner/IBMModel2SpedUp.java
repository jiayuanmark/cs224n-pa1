package cs224n.wordaligner;

import cs224n.util.CounterMap;
import cs224n.util.Counters;
import java.lang.Math;
import java.util.List;

public class IBMModel2SpedUp implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double DELTA = 0.01;
	private static final double EPS = 1e-10;
	//private static final double LEAK = 0.08; // french
	//private static final double LAMBDA = 0.01; // french
	private static final double LAMBDA = 0.01; // hindi
	private static final double LEAK = 0.08; // hindi
	

	
	// target conditioned by source
	private CounterMap<String, String> conditionalCounter;
	private CounterMap<Integer, Integer> positionCounter;
	
	public IBMModel2SpedUp() {
		conditionalCounter = new CounterMap<String, String>();
		positionCounter = new CounterMap<Integer, Integer>();
	}
	
	
	// cantor mapping hash (n, m, i) into one integer
	private static final int hash(int n, int m) {
		return (n + m) * (n + m +1) / 2 + m;
	}
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment align = new Alignment();
		int numSrcWords = sentencePair.getSourceWords().size();
		int numTgtWords = sentencePair.getTargetWords().size();
		List<String> srcWords = sentencePair.getSourceWords();
		List<String> tgtWords = sentencePair.getTargetWords();
		
		
		int length_key = hash(numTgtWords, numSrcWords);
		for (int tgtIdx = 0; tgtIdx < numTgtWords; ++tgtIdx) {
			// Initialize with a null alignment
			int key = hash(length_key, tgtIdx);
			
			double score = conditionalCounter.getCount(NULL_WORD, tgtWords.get(tgtIdx))
					* positionCounter.getCount(key, Integer.valueOf(-1));
			score = Double.isNaN(score) ? 0.0 : score;
			int maxIdx = -1;
			
			for (int srcIdx = 0; srcIdx < numSrcWords; ++srcIdx) {
				double delta = conditionalCounter.getCount(srcWords.get(srcIdx), tgtWords.get(tgtIdx))
						* positionCounter.getCount(key, Integer.valueOf(srcIdx));
				
				if (delta >= score) {
					score = delta;
					maxIdx = srcIdx;
				}
			}
			
			// Drop null alignment
			if (maxIdx != -1)
				align.addPredictedAlignment(tgtIdx, maxIdx);
		}
		return align;
	}

	private void initialize(List<SentencePair> trainingData) {
		// Position probabilities
		for (SentencePair pair : trainingData) {
			int n = pair.getTargetWords().size();
			int m = pair.getSourceWords().size();
			
			int length_key = hash(n, m);
			for (int i = 0; i < n; i++) {
				int key = hash(length_key, i);
				
				// Calculate partition function in O(1) time
				int j_lower = (int)Math.floor(((double)(i+1)) / n * m);
				int j_upper = j_lower + 1;
				
				double knot_lower = Math.exp(-LAMBDA * Math.abs(((double)(i+1))/n - ((double)(j_lower))/m));
				double knot_upper = Math.exp(-LAMBDA * Math.abs(((double)(i+1))/n - ((double)(j_upper))/m));
				double step = Math.exp(-LAMBDA / m);
				
				double partition = 	knot_lower * (Math.pow(step, j_lower) - 1.0) / (step - 1.0) + 
									knot_upper * (Math.pow(step, m - j_upper) - 1.0) / (step - 1.0);
				
				for (int j = 0; j < m; j++) {
					double val = Math.exp(-LAMBDA * Math.abs(((double)(i+1))/n-((double)(j+1))/m));
					positionCounter.incrementCount(key, Integer.valueOf(j), (1.0-LEAK) * val / partition);
				}
				positionCounter.incrementCount(key, Integer.valueOf(-1), LEAK);
			}
		}
		positionCounter = Counters.conditionalNormalize(positionCounter);
		
		// Word probabilities
		IBMModel1 model = new IBMModel1();
		model.train(trainingData);
		conditionalCounter = model.getConditionalCounter();
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		initialize(trainingData);
		System.out.println("Finish Model 1 initialization.");
		
		CounterMap<String, String> currentConditionalCounter;
		CounterMap<Integer, Integer> currentPositionCounter;
		
		for (int iter = 1; iter <= 300; iter++) {
			
			if (iter % 100 == 0) System.out.println("Iteration: " + iter);
			
			currentConditionalCounter = new CounterMap<String, String>();
			currentPositionCounter = new CounterMap<Integer, Integer>();
			
			// For each of the sentence pair
			double sum, delta;
			for (SentencePair pair : trainingData) {
				int n = pair.getTargetWords().size();
				int m = pair.getSourceWords().size();
				
				List<String> e = pair.getTargetWords();
				List<String> f = pair.getSourceWords();
				
				int s_pair = hash(n, m);				
				
				for (int i = 0; i < n; i++) {
					int p = hash(s_pair, i);
					
					// Denominator
					sum = 0.0;
					for (int j = 0; j < m; j++) {
						sum += conditionalCounter.getCount(f.get(j), e.get(i))
								* positionCounter.getCount(p, Integer.valueOf(j));
					}
					sum += conditionalCounter.getCount(NULL_WORD, e.get(i))
							* positionCounter.getCount(p, Integer.valueOf(-1));
					
					// Get rid of NaN
					if (Math.abs(sum) < EPS) continue;
					
					// Probabilistic counts 
					for (int j = 0; j < m; j++) {
						delta = conditionalCounter.getCount(f.get(j), e.get(i))
								* (positionCounter.getCount(p, Integer.valueOf(j)) / sum);
						delta = Double.isNaN(delta) ? 0.0 : delta;
						currentConditionalCounter.incrementCount(f.get(j), e.get(i), delta);
						currentPositionCounter.incrementCount(p, Integer.valueOf(j), delta);
					}
					
					// NULL word
					delta = conditionalCounter.getCount(NULL_WORD, e.get(i))
							* (positionCounter.getCount(p, Integer.valueOf(-1)) / sum);
					delta = Double.isNaN(delta) ? 0.0 : delta;
					
					currentConditionalCounter.incrementCount(NULL_WORD, e.get(i), delta);
					currentPositionCounter.incrementCount(p, Integer.valueOf(-1), delta);
				}
			}
			
			// Normalize to get p(e_i | f_j)	
			currentConditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			currentPositionCounter = Counters.conditionalNormalize(currentPositionCounter);
			
			// Debug
			System.out.println("Iteration: " + iter);
			System.out.println(conditionalCounter.compareCounter(currentConditionalCounter));
			System.out.println(positionCounter.compareCounter(currentPositionCounter));
			
			
			// Check convergence
			if (conditionalCounter.compareCounter(currentConditionalCounter) < DELTA &&
				positionCounter.compareCounter(currentPositionCounter) < DELTA)
				break;
			
			// Update parameters
			conditionalCounter = currentConditionalCounter;
			positionCounter = currentPositionCounter;
		}

		//System.out.println("***********************");
		//System.out.println(positionCounter.toString());
		//System.out.println("***********************");
	}
}
