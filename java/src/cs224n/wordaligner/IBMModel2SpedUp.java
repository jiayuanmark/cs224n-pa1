package cs224n.wordaligner;

import cs224n.util.CounterMap;
import cs224n.util.Counters;
import java.lang.Math;
import java.util.List;

public class IBMModel2SpedUp implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double min_delta = 0.01;

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
			for (SentencePair pair : trainingData) {
				int n = pair.getTargetWords().size();
				int m = pair.getSourceWords().size();
				
				List<String> e = pair.getTargetWords();
				List<String> f = pair.getSourceWords();
				
				int s_pair = hash(n, m);				
				
				for (int i = 0; i < n; i++) {
					int p = hash(s_pair, i);
					double sum = 0.0;
					for (int j = 0; j < m; j++) {
						sum += conditionalCounter.getCount(f.get(j), e.get(i))
								* positionCounter.getCount(p, Integer.valueOf(j));
					}
					sum += conditionalCounter.getCount(NULL_WORD, e.get(i))
							* positionCounter.getCount(p, Integer.valueOf(-1));
					//System.out.println("sum: " + sum);
					for (int j = 0; j < m; j++) {
						double delta = conditionalCounter.getCount(f.get(j), e.get(i))
								* (positionCounter.getCount(p, Integer.valueOf(j)) / sum);
						currentConditionalCounter.incrementCount(f.get(j), e.get(i), delta);
						currentPositionCounter.incrementCount(p, Integer.valueOf(j), delta);
						//System.out.println("delta: " + delta);
					}
					double delta = conditionalCounter.getCount(NULL_WORD, e.get(i))
							* (positionCounter.getCount(p, Integer.valueOf(-1)) / sum);
					currentConditionalCounter.incrementCount(NULL_WORD, e.get(i), delta);
					//System.out.println("delta: " + delta);
					currentPositionCounter.incrementCount(p, Integer.valueOf(-1), delta);
				}	
			}
			
			// Normalize to get p(e_i | f_j)	
			currentConditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			currentPositionCounter = Counters.conditionalNormalize(currentPositionCounter);
			
			if (	iter % 100 == 0 &&
					conditionalCounter.compareCounter(currentConditionalCounter) < min_delta &&
					positionCounter.compareCounter(currentPositionCounter) < min_delta)
				break;
			conditionalCounter = currentConditionalCounter;
			positionCounter = currentPositionCounter;			
		}
	}
}
