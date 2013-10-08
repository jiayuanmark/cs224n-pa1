package cs224n.wordaligner;

import cs224n.util.CounterMap;
import cs224n.util.Counters;
import java.lang.Math;
import java.util.List;

public class IBMModel2SpedUp implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double DELTA = 0.01;
	private static final double EPS = 1e-10;

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
		
		int s_pair = hash(numTgtWords, numSrcWords);
		
		for (int tgtIdx = 0; tgtIdx < numTgtWords; ++tgtIdx) {
			// Initialize with a null alignment
			int p = hash(s_pair, tgtIdx);
			
			double score = conditionalCounter.getCount(NULL_WORD, tgtWords.get(tgtIdx))
					* positionCounter.getCount(p, Integer.valueOf(-1));
			int maxIdx = -1;
			
			for (int srcIdx = 0; srcIdx < numSrcWords; ++srcIdx) {
				double delta = conditionalCounter.getCount(srcWords.get(srcIdx), tgtWords.get(tgtIdx))
						* positionCounter.getCount(p, Integer.valueOf(srcIdx));
				if (delta > score) {
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
		IBMModel1 model = new IBMModel1();
		model.train(trainingData);
		conditionalCounter = model.getConditionalCounter();
		for (SentencePair pair : trainingData) {
			int n = pair.getTargetWords().size();
			int m = pair.getSourceWords().size();
			int s_pair = hash(n, m);				
			for (int i = 0; i < n; i++) {
				int p = hash(s_pair, i);
				for (int j = 0; j < m; j++) {
					if (positionCounter.getCount(p, Integer.valueOf(j)) == 0)
						positionCounter.incrementCount(p, Integer.valueOf(j), Math.random());
				}
				if (positionCounter.getCount(p, Integer.valueOf(-1)) == 0)
					positionCounter.incrementCount(p, Integer.valueOf(-1), Math.random());
			}
		}
		positionCounter = Counters.conditionalNormalize(positionCounter);
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		initialize(trainingData);
		System.out.println("Finish Model 1 initialization.");
		
		CounterMap<String, String> currentConditionalCounter;
		CounterMap<Integer, Integer> currentPositionCounter;
		
		for (int iter = 1; iter <= 2000; iter++) {
			
			if (iter % 100 == 0) System.out.println("Iter: " + iter);
			
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
					if (Math.abs(sum) < EPS) {
						System.out.print("Too small: " + sum);
						continue;
					}
					
					// Probabilistic counts 
					for (int j = 0; j < m; j++) {
						delta = conditionalCounter.getCount(f.get(j), e.get(i))
								* (positionCounter.getCount(p, Integer.valueOf(j)) / sum);
						currentConditionalCounter.incrementCount(f.get(j), e.get(i), delta);
						currentPositionCounter.incrementCount(p, Integer.valueOf(j), delta);
					}
					
					// NULL word
					delta = conditionalCounter.getCount(NULL_WORD, e.get(i))
							* (positionCounter.getCount(p, Integer.valueOf(-1)) / sum);
					
					currentConditionalCounter.incrementCount(NULL_WORD, e.get(i), delta);
					currentPositionCounter.incrementCount(p, Integer.valueOf(-1), delta);
				}	
			}
			
			// Normalize to get p(e_i | f_j)	
			currentConditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			currentPositionCounter = Counters.conditionalNormalize(currentPositionCounter);
			
			if (	iter % 100 == 0 &&
					conditionalCounter.compareCounter(currentConditionalCounter) < DELTA &&
					positionCounter.compareCounter(currentPositionCounter) < DELTA)
				break;
			conditionalCounter = currentConditionalCounter;
			positionCounter = currentPositionCounter;			
		}
	}
}
