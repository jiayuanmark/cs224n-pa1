package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.lang.Math;

import edu.stanford.nlp.mt.base.FeatureValue;
import edu.stanford.nlp.mt.base.Featurizable;
import edu.stanford.nlp.mt.base.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * Number of vowel of the rule.
 * 
 * @author Jiayuan Ma
 *
 */
public class VowelNumber implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "VowelNum";
  private static final String vowel = "aeiou";

  private static int CountVowel(String s) {
    int ret = 0;
    for (int i = 0; i < s.length(); ++i) {
      for (int j = 0; j < vowel.length(); ++j) {
        if (s.charAt(i) == vowel.charAt(j)) {
          ++ret;
          break;
        }
      }
    }
    return ret;
  }

  @Override
  public void initialize() {}

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {
    List<FeatureValue<String>> features = Generics.newLinkedList();
    
    features.add(new FeatureValue<String>(String.format("%s:%d-%d", FEATURE_NAME, CountVowel(f.sourcePhrase.toString()), CountVowel(f.targetPhrase.toString())), 1.0));

    return features;
  }
}
