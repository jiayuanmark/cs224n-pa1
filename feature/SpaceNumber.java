package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.lang.Math;

import edu.stanford.nlp.mt.base.FeatureValue;
import edu.stanford.nlp.mt.base.Featurizable;
import edu.stanford.nlp.mt.base.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.Generics;

/**
 * Number of space of the rule.
 * 
 * @author Jiayuan Ma
 *
 */
public class SpaceNumber implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "SpaceNum";
  
  @Override
  public void initialize() {}

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {
    List<FeatureValue<String>> features = Generics.newLinkedList();
    
    System.out.println(f.targetPhrase.toString() + " " + f.sourcePhrase.toString());

    return features;
  }
}
