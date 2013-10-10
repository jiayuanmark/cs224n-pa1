package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.lang.Math;

import edu.stanford.nlp.mt.base.FeatureValue;
import edu.stanford.nlp.mt.base.Featurizable;
import edu.stanford.nlp.mt.base.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * Absolute difference of the rule.
 * 
 * @author Jiayuan Ma
 *
 */
public class DimensionDifference implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "DimDiff";
  
  @Override
  public void initialize() {}

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {
    List<FeatureValue<String>> features = Generics.newLinkedList();
    
    features.add(new FeatureValue<String>(String.format("%s:%d",FEATURE_NAME, Math.abs(f.targetPhrase.size() - f.sourcePhrase.size())), 1.0));
    return features;
  }
}
