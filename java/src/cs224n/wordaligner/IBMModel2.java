package cs224n.wordaligner;

import cs224n.util.*;
import java.util.List;

public class IBMModel2 implements WordAligner {

	private static final long serialVersionUID = -6202996450784531039L;
	private static final double min_delta = 0.00001;

	// target conditioned by source
	private CounterMap<String, String> conditionalCounter;
	private CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> positionCounter;
	
	public IBMModel2() {
		conditionalCounter = new CounterMap<String, String>();
		positionCounter = new Counter<Pair<Pair<Integer, Integer>, Integer>, Integer>();
	}
	
	
	@Override
	public Alignment align(SentencePair sentencePair) {
		Alignment align = new Alignment();
		int numSrcWords = sentencePair.getSourceWords().size();
		int numTgtWords = sentencePair.getTargetWords().size();
		List<String> srcWords = sentencePair.getSourceWords();
		List<String> tgtWords = sentencePair.getTargetWords();
		
		for (int tgtIdx = 0; tgtIdx < numTgtWords; ++tgtIdx) {
			// Initialize with a null alignment
			double score = conditionalCounter.getCount(tgtWords.get(tgtIdx), NULL_WORD);
			int maxIdx = -1;
			
			for (int srcIdx = 0; srcIdx < numSrcWords; ++srcIdx) {
				if (conditionalCounter.getCount(tgtWords.get(tgtIdx), srcWords.get(srcIdx)) > score) {
					score = conditionalCounter.getCount(srcWords.get(tgtIdx), tgtWords.src(tgtIdx));
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
		//TODO randomly initialize position counter
	}

	@Override
	public void train(List<SentencePair> trainingData) {
		// For each of the sentence pair
		initialize(trainingData);
		int i = 0;
		CounterMap<String, String> currentConditionalCounter;
		CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer> currentPositionCounter;
		while (i < 100) {
			for (SentencePair pair : trainingData) {
				currentConditionalCounter = new CounterMap<String, String>();
				currentPositionCounter = new CounterMap<Pair<Pair<Integer, Integer>, Integer>, Integer>();
				// count(f_j, e_i) where j = 1, ..., m 
				Pair<Integer, Integer> s_pair = new Pair(pair.getSourceWords(), pair.getTargetWords());

				for (int i = 0; i < pair.getSourceWords().size(); i++)
					for (int j = 0; j < pair.getTargetWords().size(); j++) {
						Pair<Pair<Integer, Integer>, Integer> p = new Pair(s_pair, j);
						currentConditionalCounter.incrementCount(tgtWord, srcWord, conditionalCounter.getCount(pair.getTargetWords().get(j), pair.getSourceWords().get(i)) * positionCounter.getCount(p, i));
						currentPositionCounter.incrementCount(p, i, conditionalCounter.getCount(pair.getTargetWords().get(j), pair.getSourceWords().get(i)) * positionCounter.getCount(p, i));
				}
				// count(f_0, e_i)
				for (int j = 0; j < pair.getTargetWords.size(); j++) {
					currentConditionalCounter.incrementCount(tgtWord, NULL_WORD, conditionalCounter.getCount(pair.getTargetWords().get(j), NULL_WORD) * positionCounter.get(p, 0));
					Pair<Pair<Integer, Integer>, Integer> p = new Pair(s_pair, j);
					currentPositionCounter.incrementCount(p, 0, conditionalCounter.getCount(pair.getTargetWords().get(j), NULL_WORD) * positionCounter.get(p, 0));
				}	
		}
		
			// Normalize to get p(e_i | f_j)	
			conditionalCounter = Counters.conditionalNormalize(currentConditionalCounter);
			positionCounter = Counters.conditionalNormalize(currentPositionCounter);
			i ++;
		}
	}

}
