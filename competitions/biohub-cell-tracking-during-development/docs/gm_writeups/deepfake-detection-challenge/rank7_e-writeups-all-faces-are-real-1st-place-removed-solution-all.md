# 1st Place Removed Solution - All Faces Are Real Team

**Kaggle, Facebook Host Team and Fellow Competitors:**

First of all, we want to put on record our gratitude for Kaggle and the Facebook host team for putting the effort into creating the dataset and hosting this competition, and we give our congratulations to all eventual prize winners.

We'd like to use this statement to further explain the circumstances which led to our winning solution being voided, and our position on the LB being moved with accordance to our second solution. 

In anticipation of shake-up of the competition on the private LB, we prepared our two solutions which finished with private LB scores 0.42320 and 0.44531 respectively. For the 0.44531 solution, which scored better on the public LB, we used competition data only and an unweighted mean of 12 models: this is the solution that enabled us to retain our 7th position on the LB. For our original winning solution (0.42320) we mixed 6 models trained using competition data with 9 models trained with some additional external data (our more adventurous submission).

For our original winning model, we used the following additional data:

- **The flickrface dataset**: we used a [resized version](https://www.kaggle.com/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px) of this dataset. A few of these images had licenses which didn't allow commercial use, so in line with clarifications from Kaggle in the external data thread, we used the license information available from the original github to select and train **only** on images with license types that are acceptable for this competition ([CC-BY](https://creativecommons.org/licenses/by/2.0/),  [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/), [Public Domain CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), or [U.S. Government Works](http://www.usa.gov/copyright.shtml))

- **Youtube videos images**: we manually created a face image dataset from a handful of youtube videos with [CC-BY](https://support.google.com/youtube/answer/2797468?hl=en-GB) license, which explicitly [allows for commercial use](https://creativecommons.org/licenses/by/3.0/).

We chose these data sources with the belief that they met the rules on external data, specifically that external data must be *"available to use by all participants of the competition for purposes of the competition at no cost to the other participants"*, and the additional statements in the external data thread that they must be available for commercial use and not restricted to academics etc.

However, in our discussions with Facebook and Kaggle, we were told that despite fulfilling this we were contravening the rules on Winning Submission Documentation:

&gt; WINNING SUBMISSION DOCUMENTATION (Section 4 of the Competition-Specific Rules)
In addition to compliance with the Kaggle Documentation Guidelines at [https://www.kaggle.com/WinningModelDocumentationGuidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines), the winning submission documentation must conform with the following guidelines:

&gt; A. If any part of the submission documentation depicts, identifies, or includes any person that is not an individual participant or Team member, you must have all permissions and rights from the individual depicted, identified, or included and you agree to provide Competition Sponsor and PAI with written confirmation of those permissions and rights upon request.

&gt; B. Submission documentation must not infringe, misappropriate, or violate any rights of any third party including, without limitation, copyright (including moral rights), trademark, trade secret, patent or rights of privacy or publicity.

**Specifically, we were asked to provide "additional permissions or licenses from individuals appearing in [our] external dataset"**. Unfortunately, since the data was from public datasets, we didn't have specific written permission from each individual appearing in them, nor did we have any way of identifying these individuals. We didn't realise while competing that external data in this competition falls under 'documentation' as well as the external data rules, so we did not secure these permissions from individuals depicted above and beyond the licensing requirements. 

We suspect that most competitors also did not realise these additional restrictions existed - we are unable to find any data posted in the External Data Thread which meets this threshold with a brief scan. During the competition, the rules on external data were repeatedly clarified, so this leaves us wondering why Kaggle never took the opportunity to clarify that external data must additionally follow the more restrictive rules for winning submission documentation.

An additional concern brought to us was that **Facebook felt some of our external data "clearly appears to infringe third party rights" despite being labelled as CC-BY** (it's not clear what data they were referring to specifically). Even if this were the case, it seems unreasonable to us that a Kaggle team should have to trace and verify that someone who publishes a dataset themselves has the rights to do so, and that we should have to engage rights clearance services in order to make a competition submission - it was suggested to us that we could have run our external data past our lawyer before making our submissions.

**While we feel that these extra rules could have been made clear during the competition, and we hope that Kaggle will begin to clarify these rules in future competitions, we understand that there is little we can do in this instance.** We have had a constructive call with both Kaggle and Facebook which we thank them for. After this call, it was agreed that because we did not knowingly seek to undermine any rules, that our submission that did not use any external data should be allowed to remain and only the winning submission is to be disqualified.

That being said, we are very disappointed by this outcome after spending so many months on the competition. **Successful Kaggle competitions rely on a trust between competitors and Kaggle that the rules will be fairly explained and applied, and this trust has been damaged.** We welcome any thoughts from the community on this matter.

Giba, Mikel, Yifan, Gary and Qishen  
All Faces are Real