package cs224n.wordaligner;

import cs224n.util.*;
import java.lang.Math;
import java.util.List;

public class IBMModel2 implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double min_delta = 0.00001;

	// target conditioned by source
	private CounterMap<String, String> conditionalCounter;
	private CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> positionCounter;
	
	public IBMModel2() {
		conditionalCounter = new CounterMap<String, String>();
		positionCounter = new CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer>();
	}
	
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment align = new Alignment();
		int numSrcWords = sentencePair.getSourceWords().size();
		int numTgtWords = sentencePair.getTargetWords().size();
		List<String> srcWords = sentencePair.getSourceWords();
		List<String> tgtWords = sentencePair.getTargetWords();
		Pair<Integer, Integer> s_pair = new Pair(numTgtWords, numSrcWords);
		for (int tgtIdx = 0; tgtIdx < numTgtWords; ++tgtIdx) {
			// Initialize with a null alignment
			Pair<Pair<Integer, Integer>, Integer> p = new Pair(s_pair, Integer.valueOf(tgtIdx));
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
			List<String> e = pair.getTargetWords();
			int m = pair.getSourceWords().size();
			List<String> f = pair.getSourceWords();
			Pair<Integer, Integer> s_pair = new Pair(n, m);				
			for (int i = 0; i < n; i++) {
				Pair<Pair<Integer, Integer>, Integer> p = new Pair(s_pair, Integer.valueOf(i));
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
		// For each of the sentence pair
		initialize(trainingData);
		CounterMap<String, String> currentConditionalCounter;
		CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> currentPositionCounter;
		while (true) {
			currentConditionalCounter = new CounterMap<String, String>();
			currentPositionCounter = new CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer>();
			
			for (SentencePair pair : trainingData) {
				int n = pair.getTargetWords().size();
				List<String> e = pair.getTargetWords();
				int m = pair.getSourceWords().size();
				List<String> f = pair.getSourceWords();
				Pair<Integer, Integer> s_pair = new Pair(n, m);				
				for (int i = 0; i < n; i++) {
					Pair<Pair<Integer, Integer>, Integer> p = new Pair(s_pair, Integer.valueOf(i));
					double sum = 0.0;
					for (int j = 0; j < m; j++) {
						sum += conditionalCounter.getCount(f.get(j), e.get(i))
								* positionCounter.getCount(p, Integer.valueOf(j));
					}
					sum += conditionalCounter.getCount(NULL_WORD, e.get(i))
							* positionCounter.getCount(p, Integer.valueOf(-1));
					System.out.println("sum: " + sum);
					for (int j = 0; j < m; j++) {
						double delta = conditionalCounter.getCount(f.get(j), e.get(i))
								* (positionCounter.getCount(p, Integer.valueOf(j)) / sum);
						currentConditionalCounter.incrementCount(f.get(j), e.get(i), delta);
						currentPositionCounter.incrementCount(p, Integer.valueOf(j), delta);
						System.out.println("delta: " + delta);
					}
					double delta = conditionalCounter.getCount(NULL_WORD, e.get(i))
							* (positionCounter.getCount(p, Integer.valueOf(-1)) / sum);
					currentConditionalCounter.incrementCount(NULL_WORD, e.get(i), delta);
					System.out.println("delta: " + delta);
					currentPositionCounter.incrementCount(p, Integer.valueOf(-1), delta);
				}	
			}
			
			// Normalize to get p(e_i | f_j)	
			currentConditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			currentPositionCounter = Counters.conditionalNormalize(currentPositionCounter);
			if (conditionalCounter.compareCounter(currentConditionalCounter) < 0.01 &&
					positionCounter.compareCounter(currentPositionCounter) < 0.01)
				break;
			conditionalCounter = currentConditionalCounter;
			positionCounter = currentPositionCounter;
		}
		
	}

}
