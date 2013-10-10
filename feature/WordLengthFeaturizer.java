package edu.stanford.nlp.mt.decoder.feat;

import java.util.*;

import edu.stanford.nlp.util.*;
import edu.stanford.nlp.mt.base.*;
import edu.stanford.nlp.mt.decoder.util.*;


public class WordLengthFeaturizer implements 
    DerivationFeaturizer<IString, String>,
    RuleIsolationScoreFeaturizer<IString, String> {

  @Override
  public void initialize() { }

  @Override
  public void initialize(int sourceInputId,
                         List<ConcreteRule<IString, String>> ruleList,
                         Sequence<IString> source) {
  }

  
  @Override
  public List<FeatureValue<String>> featurize(
    Featurizable<IString, String> f) {
      List<FeatureValue<String>> features = Generics.newLinkedList();
      Derivation d = f.derivation;
      double sum = 0.0;
      double count = 0.0;
      while (d != null) {
        if (d.rule != null) {
            sum += d.rule.abstractRule.target.size();
            count += 1;
        }
        d = d.preceedingDerivation;
      }
      features.add(new FeatureValue<String>("AVGLENGTH", sum / count));
    return features;
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
    Featurizable<IString,String> f)  {
    List<FeatureValue<String>> features = Generics.newLinkedList();
    return features;
  } 
} 
