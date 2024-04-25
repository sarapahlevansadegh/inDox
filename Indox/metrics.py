import gc

import numpy as np
import torch.cuda
from bert_score import BERTScorer
from collections import defaultdict
from unieval import DialogEvaluator

Example = {
    "query": ["How did Cinderella reach her happy ending ?"],
    "answer": ['''Cinderella reached her happy ending with the help of a magical bird that granted her wishes.
    Despite being mistreated by her stepmother and stepsisters, Cinderella's kindness and hopefulness prevailed. With
    the bird's assistance, she was able to attend a royal festival where the prince fell in love with her. Through a
    series of events and magical interventions, including being dressed in beautiful attire by the bird, Cinderella
    captured the heart of the prince and eventually rode away with him, revealing her true identity and finding her
    happily ever after.'''],
    "context": [
        '''The provided documentation describes the classic fairy tale of Cinderella. In the story, Cinderella is
    mistreated by her step-sisters and step-mother, but with the help of a magical bird that grants her wishes,
    she is able to attend a royal festival where the prince falls in love with her. With the bird's help, Cinderella
    attends the festival in increasingly beautiful dresses and eventually rides away with the prince. At the end of
    the tale, the true nature of Cinderella is revealed with''',
        '''The documentation provided is a retelling of the classic fairy tale of Cinderella. It describes how
    Cinderella, a kind and beautiful young girl, is mistreated by her stepmother and stepsisters but remains hopeful
    and kind. With the help of a magical bird that fulfills her wishes, Cinderella is able to attend a royal festival
    where the prince falls in love with her. Despite her stepsisters' attempts to deceive the prince, the truth is
    revealed with the help of the magical bird, and''',
        '''The provided text seems to be an excerpt from the classic fairy tale of Cinderella. It tells the story of a
    young girl who faces mistreatment from her stepmother and stepsisters. Despite their cruelty, she remains kind
    and caring. Her father, unable to recognize Cinderella, who he believes to be an unknown maiden, tries to find
    her by cutting down a tree and later trapping her in a pigeon-house. Eventually, through a series of events,
    the prince discovers Cinderella's true identity and''',
        '''The documentation provided includes multiple excerpts detailing the classic fairy tale of Cinderella. The
    story revolves around a young girl who faces mistreatment from her stepmother and stepsisters but remains kind
    and hopeful. Despite enduring hardships and cruel treatment, Cinderella's character shines through. The narrative
    describes how Cinderella, with the help of magical elements like a little white bird and a hazel tree,
    is able to attend a royal festival and capture the heart of the prince. The tale also touches upon the challenges
    Cinder''',
        '''The provided text is an excerpt from the famous fairy tale \"Cinderella.\" It narrates the story of
        Cinderella, who is mistreated by her stepmother and stepsisters but is aided by a magical hazel tree and a
        little white bird. When her stepsisters are invited to a royal festival to find a bride for the prince,
        Cinderella is left behind. The bird helps her get dressed in beautiful attire, and she attends the festival
        where she captivates everyone, including the prince '''
    ]
}

Example2 = {
    "query": ["How did Cinderella reach her happy ending ?"],
    "answer": ['''Cinderella reached her happy ending by attending the festival in a splendid dress and golden 
    slippers provided by the bird. The king's son danced only with her, and when she tried to leave, he attempted to 
    follow her but she escaped. However, he had smeared the staircase with pitch, causing her golden slipper to get 
    stuck. The king's son found the slipper and went to Cinderella's home to find the owner. After trying it on her 
    stepsisters, he finally found Cinderella, and they were reunited. They rode away together, and the two white 
    doves confirmed her as the true bride. They lived happily ever after.'''],
    "context": [
        'grave beneath the hazel-tree, and cried -      shiver and quiver, little tree,      silver and gold throw down over me. Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk and silver.  She put on the dress with all speed, and went to the wedding.  Her step-sisters and the step-mother however did not know her, and thought she must be a foreign princess, for she looked so beautiful in the golden dress. They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes.  The prince approached her, took her by the hand and danced with her. He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner. She danced till it was evening, and then she wanted to go home.',
        "seated herself on a stool, drew her foot out of the heavy wooden shoe, and put it into the slipper, which fitted like a glove.  And when she rose up and the king's son looked at her face he recognized the beautiful maiden who had danced with him and cried, that is the true bride.  The step-mother and the two sisters were horrified and became pale with rage, he, however, took cinderella on his horse and rode away with her.  As they passed by the hazel-tree, the two white doves cried -      turn and peep, turn and peep,      no blood is in the shoe,      the shoe is not too small for her,      the true bride rides with you, and when they had cried that, the two came flying down and placed themselves on cinderella's shoulders, one on the right, the other on the left, and remained sitting there. When the wedding with the king's son was to be celebrated, the",
        "And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden.  And when she went to the festival in the dress, no one knew how to speak for astonishment.  The king's son danced with her only, and if any one invited her to dance, he said this is my partner. When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her.  The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck.  The king's son picked it up, and it was small and dainty, and all golden.  Next morning, he went with it to",
        "asked his two step-daughters what he should bring back for them. Beautiful dresses, said one, pearls and jewels, said the second. And you, cinderella, said he, what will you have.  Father break off for me the first branch which knocks against your hat on your way home.  So he bought beautiful dresses, pearls and jewels for his two step-daughters, and on his way home, as he was riding through a green thicket, a hazel twig brushed against him and knocked off his hat.  Then he broke off the branch and took it with him.  When he reached home he gave his step-daughters the things which they had wished for, and to cinderella he gave the branch from the hazel-bush.  Cinderella thanked him, went to her mother's grave and planted the branch on it, and wept so much that the tears fell down on it and watered it.  And it grew and became a handsome",
        "But the king's son said, I will go with you and bear you company, for he wished to see to whom the beautiful maiden belonged. She escaped from him, however, and sprang into the pigeon-house.  The king's son waited until her father came, and then he told him that the unknown maiden had leapt into the pigeon-house.  The old man thought, can it be cinderella.  And they had to bring him an axe and a pickaxe that he might hew the pigeon-house to pieces, but no one was inside it.  And when they got home cinderella lay in her dirty clothes among the ashes, and a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had"]
}
Example3 = {
    'query': ['how cinderella reach her happy ending ?'],
    'answer': ['Persian Gulf is surrounded by middle east countries'],
    'context': ['Persian Gulf is surrounded by middle east countries']
}


def convert_to_json(output_list, src_list=None, ref_list=None, context_list=None, scores=None, doc_id=None,
                    system_id=None):
    """
        Convert the data into the json format.

        output_list: a list of model output
        src_list: source input for different NLG tasks. For example, source document for summarization
                  and dialogue history for dialogue response generation
        ref_list: human-annotated groundtruth
        context_list: the context needed to evaluate several specific dimension. For example,
                      additional factual information when evaluating engagingness and groundedness in dialogues
        scores: human scores for evaluating the model output. They can be used to calculate the correlation
                between evaluators and human judgements. The scores should be stored in a dictionary. For example,
                {'fluency': 2.0, 'coherence': 3.0} could be the human score for a sample.
        doc_id: the index of the input source. It can be used to calculate summary-level correlation for summarzation
        system_id: the index of the generation system. It can be used to calculate system-level correlation.
    """
    json_data = []
    for i in range(len(output_list)):
        cur = {}
        cur['system_output'] = output_list[i]
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
        json_data.append(cur)
    return json_data


class BertEvaluator(BERTScorer):
    def evaluate(self, inputs, verbose=False, batch_size=64, return_hash=False):
        _, answer, context = inputs['query'], inputs['answer'], inputs['context']
        scores = defaultdict(list)
        if not isinstance(context, list) and len(context) > 1:
            context = list(context)
        if not isinstance(answer, list):
            answer = [answer]
        K = len(context)

        for i in range(K):
            P, R, F1 = self.score(answer, [context[i]], verbose=False)
            [scores[key].append(value) for key, value in [('P', P.numpy()), ('R', R.numpy()), ('F1', F1.numpy())]]
        scores['K'].append(K)
        mP, mR, mF1 = np.array(scores['P']).mean(), np.array(scores['R']).mean(), np.array(scores['F1']).mean()

        return mP, mR, mF1, K


class DilaougesScorer(DialogEvaluator):
    def evaluate(self, inputs):
        query, answer, context = inputs['query'], inputs['answer'], inputs['context']

        if not isinstance(context, list) and len(context) > 1:
            context = list(context)
        K = len(context)

        scores = defaultdict(list)
        for i in range(K):
            # Prepare data for pre-trained evaluators

            data = convert_to_json(output_list=[answer],
                                   src_list=[query], context_list=[context[i]])
            score = self.single_evaluate(data)
            [scores[key].append(item) for key, item in score[0].items()]

        # print("\n\nUni Eval Sores")
        # [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in scores.items()]
        # scores['K'].append(K)
        # gc.collect()
        # torch.cuda.empty_cache()
        return scores, K


def metrics(input):
    bert_score = BertEvaluator(model_type='bert-base-uncased')
    mP, mR, mF1, K = bert_score.evaluate(input)
    dilaouges = DilaougesScorer()
    dilaouges_scores, K = dilaouges.evaluate(input)

    return mP, mR, mF1, dilaouges_scores, K


answer = "Cinderella reached her happy ending by remaining kind and hopeful despite the mistreatment she faced from her stepmother and stepsisters. In the classic fairy tale, Cinderella's kindness and perseverance ultimately lead to her being recognized by the prince as the true bride whose foot fits into a golden slipper. With the help of magical gifts from a hazel tree planted on her mother's grave, Cinderella is able to attend the royal wedding in a beautiful dress and slippers. The prince dances only with her, recognizing her true identity and choosing her as his partner. This culmination of events leads to Cinderella finding her happy ending by marrying the prince and escaping the hardships imposed by her step-family."
context = "The provided documentation is a retelling of the classic fairy tale of Cinderella. It describes the story " \
          "of a young girl named Cinderella who faces mistreatment from her stepmother and stepsisters. After " \
          "Cinderella's mother passes away, her father remarries a woman with two wicked daughters who mistreat " \
          "Cinderella. Despite this mistreatment, Cinderella remains kind and hopeful. The story includes details " \
          "about Cinderella receiving a hazel branch from her father, which she plants on her mother's grave.", \
          "The documentation provided is a retelling of the classic fairy tale of Cinderella. In this version, " \
          "it tells the story of a rich man's wife who passes away, leaving her daughter with her last words of " \
          "advice to be good and pious. After her death, the man remarries a woman with two beautiful but wicked " \
          "daughters who mistreat the kind and pious daughter, whom they mockingly call Cinderella. Despite her " \
          "hardships, Cinderella remains hopeful and kind.  The story goes on to", 'The documentation provided is a ' \
                                                                                   'detailed retelling of the classic ' \
                                                                                   'fairy tale of Cinderella. It ' \
                                                                                   'narrates the story of a young ' \
                                                                                   'girl named Cinderella who is ' \
                                                                                   'mistreated by her stepmother and ' \
                                                                                   'stepsisters. The plot revolves ' \
                                                                                   'around a prince who is looking ' \
                                                                                   'for the true bride whose foot ' \
                                                                                   'fits into a golden slipper. ' \
                                                                                   'Despite the attempts of the ' \
                                                                                   'stepsisters to deceive the prince ' \
                                                                                   'by mutilating their feet, ' \
                                                                                   'it is Cinderella who ultimately ' \
                                                                                   'fits the slipper perfectly. The ' \
                                                                                   'prince recognizes her from a ' \
                                                                                   'previous encounter', \
          'grave beneath the hazel-tree, and cried -      shiver and quiver, little tree,      silver and gold throw ' \
          'down over me. Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk ' \
          'and silver.  She put on the dress with all speed, and went to the wedding.  Her step-sisters and the ' \
          'step-mother however did not know her, and thought she must be a foreign princess, for she looked so ' \
          'beautiful in the golden dress. They never once thought of cinderella, and believed that she was sitting at ' \
          'home in the dirt, picking lentils out of the ashes.  The prince approached her, took her by the hand and ' \
          'danced with her. He would dance with no other maiden, and never let loose of her hand, and if any one else ' \
          'came to invite her, he said, this is my partner. She danced till it was evening, and then she wanted to go ' \
          'home.', "The provided text describes an excerpt from the classic fairy tale of Cinderella. It details how " \
                   "Cinderella's father, on his way home, fulfills the wishes of his step-daughters for beautiful " \
                   "dresses, pearls, and jewels, and for Cinderella, he brings a hazel branch. Cinderella plants this " \
                   "branch on her mother's grave, where it grows into a handsome tree that provides her with gifts " \
                   "whenever she prays under it. The story continues with Cinderella's step-sisters getting ready " \
                   "Cinderella reached her happy ending by remaining kind and hopeful despite the mistreatment she " \
                   "faced from her stepmother and stepsisters. In the classic fairy tale, Cinderellas kindness and " \
                   "perseverance ultimately lead to her being recognized by the prince as the true bride whose foot " \
                   "fits into a golden slipper. With the help of magical gifts from a hazel tree planted on her " \
                   "mother's grave, Cinderella is able to attend the royal wedding in a beautiful dress and slippers. " \
                   "The prince dances only with her, recognizing her true identity and choosing her as his partner. " \
                   "This culmination of events leads to Cinderella finding her happy ending by marrying the prince " \
                   "and escaping the hardships imposed by her step-family. "

query = "How did Cinderella reach her happy ending?"
inputs = {"query": query, "answer": answer, "context": context}
if __name__ == "__main__":
    # scorer2 = DilaougesScorer()
    # scores2 = scorer2.evaluate(Example2)
    # scorer1 = BertEvaluator(model_type='bert-base-uncased')
    # # scorer1 = BertEvaluator(model_type='microsoft/deberta-xlarge-mnli')
    # scores1 = scorer1.evaluate(Example2)

    # mP, mR, mF1, K = metrics(inputs)
    mP, mR, mF1, dilaouges_scores, K = metrics(inputs)
    print("\n\nUni Eval Sores")
    [print(f"   {key}@{K}: {np.array(value).mean():4f}") for key, value in dilaouges_scores.items()]
    print(f"BertScore scores:\n   Precision@{K}: {mP:.4f}\n   Recall@{K}: {mR:.4f}\n   F1@{K}: {mF1:.4f}")