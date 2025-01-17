/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

////Type of word misspelled or correct
//enum class word_type : int8_t {correct = 0, misspelled = 1};

struct entry {
  std::string word;
  int64_t count;
  entry_type type;
//  word_type wType;    //Misspelled or correct
  //subwords contains indices of subwords in the Embedding Matrix
  std::vector<int32_t> subwords;
};

class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const;
  int32_t find(const std::string&, uint32_t h) const;
  void initTableDiscard();
  void initNgrams();
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

  std::shared_ptr<Args> args_;
  // Mapping of hashes of the words to its position in words_
  std::vector<int32_t> word2int_;         //id of words
  // Contains entries of all the unique words
  std::vector<entry> words_;              //words stored as entries

  std::vector<real> pdiscard_;

  //Vocabulary size of Misspelled Dataset
  int32_t misspelled_size_;

  int32_t size_;
  int32_t nwords_;                        //Number of words
  int32_t nlabels_;
  int64_t ntokens_;

  int64_t pruneidx_size_;
  std::unordered_map<int32_t, int32_t> pruneidx_;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  int32_t nwords() const;

  // Function for returning the misspelled_size_
  int32_t misspelled_words() const;

  int32_t nlabels() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&) const;
  int32_t getId(const std::string&, uint32_t h) const;
  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;
  bool discard(int32_t, real) const;
  std::string getWord(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t) const;
  const std::vector<int32_t> getSubwords(const std::string&) const;
  void getSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>&) const;

  //Getting Misspelled WOrds gven the id of the word
//  const std::vector<int32_t>& getMisspelledWords(int32_t) const;


  // Added the bool of the misspelled type
  void computeSubwords(
      const std::string&,
      std::vector<int32_t>&,
//      word_type wType,
      std::vector<std::string>* substrings = nullptr) const;
  uint32_t hash(const std::string& str) const;

  // Added bool for misspelled word in add
  void add(const std::string&);//, bool isCorrect);
  bool readWord(std::istream&, std::string&) const;

  //Modified the function to also take misspelled data as input
  void readFromFile(std::istream& ifs, std::istream& ims);

  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&)
      const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  void threshold(int64_t, int64_t);
  void prune(std::vector<int32_t>&);
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
