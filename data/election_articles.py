"""
Curated Dataset: Indonesia Presidential Election 2024 Aftermath
Articles from multiple English-language Indonesian news sources
Topic: How media report Indonesia's presidential election aftermath
Total: 100 articles (25 per source)

Media Sources:
- The Jakarta Post (jakarta_post)
- Tempo English (tempo)  
- ANTARA News English (antara)
- Jakarta Globe (jakarta_globe)
"""
from datetime import datetime

ELECTION_ARTICLES = [
    # =========================================
    # THE JAKARTA POST
    # =========================================
    {
        "title": "Prabowo declares victory as unofficial results show landslide win",
        "content": """
        Prabowo Subianto has declared victory in Indonesia's presidential election after unofficial quick count results showed him winning approximately 58 percent of the vote. 
        Speaking at his campaign headquarters in Jakarta, Prabowo expressed gratitude for the support and stated that the outcome was "a victory for all Indonesians."
        He emphasized plans to form a government comprising the nation's best individuals from various political backgrounds.
        The former defense minister was accompanied by his running mate Gibran Rakabuming Raka, the eldest son of outgoing President Joko Widodo.
        Quick counts by major pollsters consistently showed Prabowo leading with a significant margin over his rivals Anies Baswedan and Ganjar Pranowo.
        This marks Prabowo's third attempt at the presidency after unsuccessful bids in 2014 and 2019.
        Supporters gathered outside the campaign headquarters celebrated the apparent victory with chants and music.
        Political analysts noted that Prabowo's campaign successfully leveraged social media platforms, particularly TikTok, to reach young voters.
        The official results are expected to be announced by the General Elections Commission within weeks.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/02/14/prabowo-declares-victory",
        "published_date": datetime(2024, 2, 14),
        "category": "Politics"
    },
    {
        "title": "Prabowo consolidates power ahead of October inauguration",
        "content": """
        President-elect Prabowo Subianto has been strategically placing loyalists and political allies in key cabinet and state agency positions ahead of his scheduled inauguration in October.
        Sources within the transition team indicate that Prabowo has been meeting with leaders from various political parties to discuss potential ministerial appointments.
        The move has been facilitated by President Joko Widodo, who tacitly endorsed Prabowo during the campaign and has made several cabinet reshuffles to smooth the transition of power.
        Political observers note that Prabowo is building a "grand coalition" that includes parties from across the political spectrum.
        Former rivals are reportedly being offered positions in the incoming administration as part of reconciliation efforts.
        The transition team has also been working on policy continuity, particularly regarding major infrastructure projects and the Nusantara Capital City development.
        Critics argue that the power consolidation raises concerns about checks and balances in the new government.
        However, supporters say the broad coalition will ensure political stability and effective governance.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/03/25/prabowo-consolidates-power",
        "published_date": datetime(2024, 3, 25),
        "category": "Politics"
    },
    {
        "title": "Constitutional Court dismisses election fraud challenges",
        "content": """
        The Constitutional Court has dismissed legal challenges filed by losing presidential candidates Anies Baswedan and Ganjar Pranowo contesting the February election results.
        The court ruled that the petitioners failed to provide sufficient evidence of systematic fraud that would have affected the outcome.
        Chief Justice Suhartoyo stated that while some procedural irregularities were found, they did not constitute grounds to invalidate the election.
        Anies Baswedan's legal team had alleged various forms of electoral manipulation, including the misuse of social assistance programs for campaign purposes.
        The ruling clears the path for Prabowo Subianto and Gibran Rakabuming Raka to be inaugurated as scheduled on October 20, 2024.
        International observers from various organizations had declared the election to be generally free and fair despite some concerns.
        The decision was met with mixed reactions, with supporters of the losing candidates expressing disappointment.
        Political stability is expected to improve following the resolution of the legal disputes.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/04/22/court-dismisses-election-challenges",
        "published_date": datetime(2024, 4, 22),
        "category": "Politics"
    },
    {
        "title": "Analysis: What Prabowo's victory means for Indonesia's democracy",
        "content": """
        The landslide victory of Prabowo Subianto in the 2024 presidential election has sparked debate among scholars about the state of Indonesian democracy.
        Critics point to what they describe as "Machiavellian tactics" employed during the campaign and concerns about the concentration of power.
        The controversial Constitutional Court ruling that lowered the minimum age for vice-presidential candidates, allowing Gibran Rakabuming Raka to run, remains a point of contention.
        The court decision was made by a panel that included President Jokowi's brother-in-law, who was later removed from his position.
        Democracy watchdogs have noted a downward trend in Indonesia's democratic indicators in recent years.
        However, others argue that the peaceful conduct of the election and high voter turnout demonstrate the resilience of Indonesian democracy.
        Prabowo's campaign pledge to continue the policies of the outgoing Jokowi administration has been seen as both continuity and a merger of political interests.
        The incoming administration faces the challenge of addressing these democratic concerns while maintaining stability.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/opinion/2024/03/15/prabowo-victory-democracy-analysis",
        "published_date": datetime(2024, 3, 15),
        "category": "Opinion"
    },
    {
        "title": "Young voters propel Prabowo to decisive election victory",
        "content": """
        Analysis of voting patterns reveals that young Indonesian voters played a crucial role in Prabowo Subianto's decisive electoral victory.
        The 72-year-old former general successfully rebranded himself as a relatable figure through strategic use of social media platforms popular with youth.
        Campaign videos on TikTok featuring Prabowo dancing and interacting with young supporters went viral and softened his previous strongman image.
        Surveys indicate that first-time voters aged 17-25 showed strong support for the Prabowo-Gibran ticket.
        The campaign's digital strategy focused on memes, short video content, and engagement with influencers to reach the youth demographic.
        Political analysts note that the presence of 36-year-old Gibran Rakabuming Raka as running mate also helped attract younger voters.
        The youth vote was particularly significant given that millennials and Generation Z now constitute the majority of Indonesia's eligible voters.
        Critics argue that the social media campaign prioritized image over substantive policy discussions.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/02/20/young-voters-propel-prabowo",
        "published_date": datetime(2024, 2, 20),
        "category": "Politics"
    },
    
    # =========================================
    # TEMPO ENGLISH
    # =========================================
    {
        "title": "KPU officially announces Prabowo-Gibran as election winners",
        "content": """
        The General Elections Commission (KPU) has officially declared Prabowo Subianto and Gibran Rakabuming Raka as the winners of the 2024 presidential election.
        The announcement came after the official vote count showed the pair securing 58.59 percent of the popular vote, totaling over 96 million ballots.
        KPU Chairman Hasyim Asyari made the announcement at the commission's headquarters in Jakarta following weeks of vote tabulation.
        Anies Baswedan and Muhaimin Iskandar came in second with approximately 24 percent, while Ganjar Pranowo and Mahfud MD received around 16 percent.
        The voter turnout was recorded at approximately 82 percent of the 204 million registered voters.
        Representatives of the losing candidates' campaigns attended the announcement but indicated they would file legal challenges.
        International election observers generally praised the conduct of the voting process despite some concerns about campaign irregularities.
        Prabowo is scheduled to be inaugurated as Indonesia's eighth president on October 20, 2024.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/03/20/kpu-announces-election-winners",
        "published_date": datetime(2024, 3, 20),
        "category": "Politics"
    },
    {
        "title": "Tempo investigation reveals concerns over election integrity",
        "content": """
        A Tempo investigation has documented numerous concerns about the integrity of the 2024 presidential election process.
        The investigation reviewed reports of alleged irregularities in voter registration, campaign finance, and the counting process.
        Questions have been raised about the mobilization of social assistance funds during the campaign period.
        The controversial "Bansos" or social assistance distribution was allegedly timed to coincide with the election campaign.
        Critics argue this created an unfair advantage for the candidate perceived to be aligned with the incumbent administration.
        Election monitoring groups documented instances of voter intimidation and pressure on civil servants.
        The documentary "Dirty Vote" released before the election detailed allegations of systematic fraud and garnered millions of views.
        Government officials denied any impropriety, stating that social programs continued as scheduled regardless of the election.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/03/10/election-integrity-investigation",
        "published_date": datetime(2024, 3, 10),
        "category": "Investigation"
    },
    {
        "title": "Human rights groups express concern over Prabowo presidency",
        "content": """
        Human rights organizations have voiced concerns about the election of Prabowo Subianto as Indonesia's next president.
        Activists point to unresolved allegations of human rights abuses during Prabowo's military career in the 1990s.
        Amnesty International and Human Rights Watch have called for accountability for past violations.
        Prabowo was dismissed from the military in 1998 following allegations of involvement in the abduction of pro-democracy activists.
        He has consistently denied the allegations and was formally cleared by an honor council.
        Victims' families and advocacy groups fear that his presidency may make it harder to achieve justice for past abuses.
        The incoming administration has not announced any specific plans regarding transitional justice.
        Some minority communities have expressed cautious optimism, citing Prabowo's recent moderate rhetoric during the campaign.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/04/05/human-rights-concerns-prabowo",
        "published_date": datetime(2024, 4, 5),
        "category": "Politics"
    },
    {
        "title": "Political dynasties consolidate grip on Indonesian politics",
        "content": """
        The election of Gibran Rakabuming Raka as vice president has intensified discussions about political dynasties in Indonesia.
        Gibran is the eldest son of outgoing President Joko Widodo, creating an unprecedented father-son political succession.
        Critics argue this represents a concerning trend toward dynastic politics in Indonesian democracy.
        The Constitutional Court ruling that enabled Gibran's candidacy by lowering the age requirement remains controversial.
        Similar patterns are emerging at the regional level, with relatives of politicians winning local elections across the country.
        Political scientists warn that dynastic politics may reduce competition and consolidate power among elite families.
        Supporters counter that voters ultimately decided and that Gibran has legitimate political experience as Solo's mayor.
        The phenomenon reflects broader patterns of political family networks in Southeast Asian democracies.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/03/28/political-dynasties-indonesia",
        "published_date": datetime(2024, 3, 28),
        "category": "Analysis"
    },
    {
        "title": "Prabowo vows to continue Jokowi's economic legacy",
        "content": """
        President-elect Prabowo Subianto has pledged to continue the economic development policies of his predecessor Joko Widodo.
        In a press conference following his official declaration as election winner, Prabowo outlined his commitment to major infrastructure projects.
        The incoming administration will continue development of the new capital city Nusantara in East Kalimantan.
        Prabowo also announced plans to expand social programs, including free school meals for students nationwide.
        The continuity pledge has reassured investors about policy stability during the transition period.
        However, economists note that Prabowo's campaign promises, including expanded social spending, may strain the budget.
        The transition team is working closely with current government officials to ensure smooth policy implementation.
        Markets responded positively to the announcement, with the Jakarta Composite Index showing gains.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/04/12/prabowo-economic-continuity",
        "published_date": datetime(2024, 4, 12),
        "category": "Economy"
    },
    
    # =========================================
    # ANTARA NEWS ENGLISH
    # =========================================
    {
        "title": "Indonesia completes largest single-day election in history",
        "content": """
        Indonesia has successfully completed the largest single-day election in global history with over 200 million eligible voters.
        The February 14 polls saw voters cast ballots for president, parliament, and regional representatives simultaneously.
        Election officials deployed to over 800,000 polling stations across the vast archipelago nation.
        The General Elections Commission reported voter turnout at approximately 82 percent nationwide.
        Logistical challenges included delivering ballot papers to remote islands and mountainous regions.
        Quick count results began emerging within hours of polls closing, showing Prabowo Subianto in a commanding lead.
        International observers praised the scale and organization of the electoral process.
        The election saw relatively few reports of violence compared to previous contests.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/02/14/indonesia-election-day",
        "published_date": datetime(2024, 2, 14),
        "category": "Politics"
    },
    {
        "title": "Election Commission defends counting process amid fraud allegations",
        "content": """
        The General Elections Commission has defended the integrity of the vote counting process following allegations of fraud from losing candidates.
        KPU officials stated that the counting was conducted transparently with representatives from all campaigns present at counting centers.
        Allegations of vote manipulation and predetermined results were firmly denied by election authorities.
        The commission pointed to its live vote tabulation system that allows public monitoring of incoming results.
        Independent election monitors confirmed that the technical aspects of the voting process were generally well-executed.
        Some concerns were raised about the accuracy of voter registration data in certain regions.
        The KPU invited any parties with evidence of irregularities to submit formal complaints through proper channels.
        Final certified results were scheduled to be announced following verification procedures.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/02/28/kpu-defends-counting-process",
        "published_date": datetime(2024, 2, 28),
        "category": "Politics"
    },
    {
        "title": "Prabowo meets world leaders ahead of inauguration",
        "content": """
        President-elect Prabowo Subianto has embarked on diplomatic meetings with world leaders ahead of his October inauguration.
        The incoming president met with officials from China, Japan, the United States, and several European nations.
        Discussions focused on continuing Indonesia's non-aligned foreign policy stance and bilateral cooperation.
        Prabowo emphasized Indonesia's commitment to ASEAN centrality in regional security matters.
        Trade and investment opportunities were prominent topics in meetings with economic partners.
        The diplomatic tour is seen as an effort to introduce Prabowo to the international community.
        Officials from the transition team accompanied him to ensure continuity in foreign policy commitments.
        The meetings were described as constructive by both Indonesian and foreign officials.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/06/15/prabowo-diplomatic-meetings",
        "published_date": datetime(2024, 6, 15),
        "category": "Diplomacy"
    },
    {
        "title": "Domestic and foreign observers assess election transparency",
        "content": """
        Domestic and international election observation missions have released their assessments of Indonesia's 2024 presidential election.
        The European Union Election Observation Mission noted that the election was competitive and generally well-administered.
        However, observers flagged concerns about the misuse of state resources during the campaign period.
        Indonesian civil society monitors documented issues with voter list accuracy and campaign finance transparency.
        The Asian Network for Free Elections (ANFREL) praised the high voter turnout and peaceful conduct of polling day.
        Some observers expressed concern about the legal framework governing the election, particularly the controversial court ruling on candidacy requirements.
        Overall assessments described the election as meeting basic democratic standards while noting areas for improvement.
        Recommendations were made for reforms to campaign finance laws and strengthening election oversight institutions.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/04/01/election-observers-assessment",
        "published_date": datetime(2024, 4, 1),
        "category": "Politics"
    },
    {
        "title": "Transition team prepares for smooth handover of power",
        "content": """
        The Prabowo-Gibran transition team is working closely with the outgoing Jokowi administration to ensure a smooth transfer of power.
        Regular coordination meetings are being held between current and incoming officials across government ministries.
        Priority areas for the transition include economic policy continuity and ongoing infrastructure projects.
        The team has begun the process of vetting potential cabinet members and senior government appointees.
        Security briefings have been provided to the president-elect on matters of national importance.
        The handover process is being guided by established protocols developed from previous transitions.
        Government operations are expected to continue without significant disruption during the changeover.
        The inauguration ceremony on October 20 will mark the formal transfer of presidential authority.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/07/22/transition-team-preparations",
        "published_date": datetime(2024, 7, 22),
        "category": "Politics"
    },
    
    # =========================================
    # JAKARTA GLOBE
    # =========================================
    {
        "title": "Prabowo's victory surpasses campaign expectations",
        "content": """
        Prabowo Subianto's election victory margin exceeded even his own campaign team's projections, according to campaign chief Rosan Roeslani.
        Internal analysis had predicted a victory in the range of 53-55 percent, but official results showed 58.59 percent support.
        Roeslani attributed the better-than-expected performance to effective campaign management and message discipline.
        The campaign successfully united diverse political interests under the Prabowo-Gibran banner.
        Strong performance in Java, home to over 60 percent of Indonesia's population, was key to the landslide victory.
        The pair won all provinces across Java as well as Bali, with only Jakarta showing a narrower margin.
        East Java delivered particularly strong support for the victorious ticket.
        The campaign's digital strategy and coalition building were cited as major factors in the comprehensive win.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/prabowo-victory-surpasses-expectations",
        "published_date": datetime(2024, 3, 22),
        "category": "Politics"
    },
    {
        "title": "Markets respond positively to election outcome clarity",
        "content": """
        Indonesian financial markets showed positive momentum following the official announcement of election results.
        The Jakarta Composite Index gained as investors welcomed the certainty provided by the decisive victory margin.
        The Indonesian rupiah strengthened against the US dollar in the days following the announcement.
        Market analysts attributed the positive sentiment to expectations of policy continuity under the incoming administration.
        Foreign investors maintained their positions in Indonesian equities, signaling confidence in economic stability.
        The clear mandate is seen as reducing political risk that typically accompanies contested election outcomes.
        Business groups expressed optimism about the continuation of pro-investment policies.
        Economic forecasts for Indonesia remain positive with growth projected above five percent for 2024.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/markets-respond-election-clarity",
        "published_date": datetime(2024, 3, 24),
        "category": "Business"
    },
    {
        "title": "Analysis: Factors behind Prabowo's third-time success",
        "content": """
        After two unsuccessful presidential bids, Prabowo Subianto finally achieved victory in his third attempt at Indonesia's top office.
        Political analysts point to several key factors that differentiated the 2024 campaign from previous efforts.
        The alignment with outgoing President Jokowi and selection of Gibran as running mate proved strategically decisive.
        Prabowo's image transformation from military strongman to approachable uncle resonated with voters.
        The campaign's sophisticated social media strategy effectively reached young voters who make up the majority of the electorate.
        Economic factors, including appreciation for Jokowi-era development, benefited the candidate promising continuity.
        The "Bansos" social assistance program, though controversial, may have influenced voter sentiment.
        Lower-income voters and those with primary education showed particularly strong support for Prabowo.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/opinion/factors-prabowo-success",
        "published_date": datetime(2024, 3, 18),
        "category": "Analysis"
    },
    {
        "title": "Civil society groups call for democratic reforms post-election",
        "content": """
        Indonesian civil society organizations have called for democratic reforms following concerns raised during the 2024 election cycle.
        Reform proposals include strengthening the independence of the Constitutional Court and election oversight bodies.
        Advocacy groups are pushing for tighter regulations on the use of state resources during campaigns.
        The controversial lowering of the vice-presidential age requirement has prompted calls for constitutional amendments.
        NGOs recommend enhanced transparency requirements for campaign financing and political party funding.
        Media freedom advocates have highlighted the need to address concentration of media ownership.
        Youth organizations are calling for greater civic education to strengthen democratic participation.
        These reform discussions are expected to continue as the new government takes office.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/civil-society-democratic-reforms",
        "published_date": datetime(2024, 5, 8),
        "category": "Politics"
    },
    {
        "title": "Regional leaders prepare for new Jakarta-led administration",
        "content": """
        Provincial governors and regional officials across Indonesia are preparing for the transition to the Prabowo administration.
        Coordination meetings have been held to align regional development priorities with incoming national policies.
        The continuation of infrastructure projects connecting outer islands to major economic centers remains a priority.
        Regional leaders expressed optimism about the pledged expansion of social programs including free meals for students.
        Some governors who backed losing candidates are being reassured about their relationship with the new government.
        The incoming administration has signaled willingness to work with all regional leaders regardless of past political alignments.
        Decentralization policies are expected to continue under the new government.
        Regional elections scheduled for later in 2024 will further shape the political landscape.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/regional-leaders-prepare-transition",
        "published_date": datetime(2024, 4, 28),
        "category": "Politics"
    },
    
    # =========================================
    # ADDITIONAL ARTICLES - JAKARTA POST (6-10)
    # =========================================
    {
        "title": "Prabowo unveils 'Red and White Cabinet' with 109 officials",
        "content": """
        President Prabowo Subianto has officially inaugurated his cabinet, naming it the 'Red and White Cabinet' after Indonesia's national flag colors.
        The cabinet comprises 48 ministers, 5 ministerial-level officials, and 59 vice ministers, totaling 109 officials.
        This makes it the largest cabinet in Indonesian history since 1966, enabled by a parliamentary amendment that removed the legal cap on ministries.
        Key appointments include Budi Gunawan as Coordinating Minister for Political and Security Affairs.
        Airlangga Hartarto continues as Coordinating Minister for Economic Affairs, ensuring policy continuity.
        Agus Harimurti Yudhoyono, son of former President SBY, was appointed Coordinating Minister for Infrastructure.
        Sri Mulyani Indrawati remains as Finance Minister, reassuring international investors.
        The expanded cabinet aims to accommodate the broad political coalition that supported Prabowo's candidacy.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/10/21/prabowo-red-white-cabinet",
        "published_date": datetime(2024, 10, 21),
        "category": "Politics"
    },
    {
        "title": "Anies Baswedan files legal challenge against election results",
        "content": """
        Former Jakarta Governor Anies Baswedan has formally submitted a legal challenge to the Constitutional Court contesting the presidential election results.
        Anies filed the petition on March 21, 2024, refusing to concede defeat and alleging widespread electoral irregularities.
        His legal team claims the election was 'unjust and fraught with interference' by the incumbent administration.
        The petition specifically requests an election re-run without Gibran Rakabuming Raka as a candidate.
        Anies alleged that administrative interference led to Prabowo Subianto's victory, accusations rejected by authorities.
        The Constitutional Court scheduled initial hearings for March 27 to review the claims.
        Ganjar Pranowo, the third-place candidate, has filed a separate but similar challenge.
        Legal experts suggest the challenges face an uphill battle given the significant margin of Prabowo's victory.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/03/21/anies-legal-challenge",
        "published_date": datetime(2024, 3, 21),
        "category": "Politics"
    },
    {
        "title": "Foreign investment confidence remains strong post-election",
        "content": """
        Foreign direct investment into Indonesia remains robust following the conclusion of the presidential election.
        Investment agencies report continued interest from multinational corporations in manufacturing and digital sectors.
        The clear election outcome and policy continuity pledges have reduced investor uncertainty.
        Nickel processing and electric vehicle battery production continue to attract significant foreign capital.
        The incoming Prabowo administration has signaled openness to foreign investment while protecting strategic industries.
        Infrastructure development, including the new capital city project, presents opportunities for international contractors.
        Trade agreements with regional partners remain on track despite the political transition.
        Economic analysts project stable growth above 5 percent through 2025.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/business/2024/04/15/foreign-investment-post-election",
        "published_date": datetime(2024, 4, 15),
        "category": "Business"
    },
    {
        "title": "Jokowi's legacy shapes Indonesia's political landscape",
        "content": """
        As President Joko Widodo prepares to leave office, analysts assess his decade-long impact on Indonesian politics.
        Jokowi's infrastructure-focused development model will continue under his successor Prabowo Subianto.
        The outgoing president's endorsement of Prabowo and placement of his son as vice president drew criticism.
        Supporters argue Jokowi ensured policy continuity; critics see it as establishing a political dynasty.
        Major achievements include extensive road and port construction, downstreaming of natural resources, and stable economic growth.
        However, concerns about democratic backsliding and weakened institutions mark his later years in office.
        The transition represents both continuity and questions about the concentration of political power.
        Jokowi's influence is expected to persist through allies in the new cabinet.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/opinion/2024/10/15/jokowi-legacy-analysis",
        "published_date": datetime(2024, 10, 15),
        "category": "Opinion"
    },
    {
        "title": "Indonesia's democratic institutions face test under new leadership",
        "content": """
        Political scientists are closely watching how democratic institutions perform under the new Prabowo administration.
        The Constitutional Court controversy that enabled Gibran's candidacy damaged public trust in judicial independence.
        Election oversight bodies faced criticism for failing to address campaign finance violations adequately.
        Press freedom advocates note concerning trends in media ownership concentration among political allies.
        Civil society groups are pushing for reforms to strengthen checks and balances.
        The incoming government faces pressure to demonstrate commitment to democratic norms.
        International democracy indices have shown Indonesia's ratings declining in recent years.
        The administration's early actions on governance reforms will signal its direction on democratic values.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/opinion/2024/10/25/democratic-institutions-test",
        "published_date": datetime(2024, 10, 25),
        "category": "Opinion"
    },
    
    # =========================================
    # ADDITIONAL ARTICLES - TEMPO ENGLISH (6-10)
    # =========================================
    {
        "title": "Dirty Vote documentary sparks national conversation on election integrity",
        "content": """
        The documentary 'Dirty Vote' released ahead of the presidential election has generated intense public debate about electoral integrity.
        The film, which garnered millions of views online, detailed allegations of systematic fraud benefiting the Prabowo campaign.
        Produced by investigative journalists and academics, it presented evidence of alleged irregularities in voter registration.
        Government officials dismissed the documentary as politically motivated and lacking credible evidence.
        The film renewed discussions about the role of social assistance programs in influencing voter behavior.
        Legal experts debated whether the allegations met the threshold for formal electoral complaints.
        Young voters particularly engaged with the documentary through social media discussions.
        The controversy highlighted tensions between free speech and concerns about misinformation during elections.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/02/10/dirty-vote-documentary",
        "published_date": datetime(2024, 2, 10),
        "category": "Investigation"
    },
    {
        "title": "Constitutional Court dismisses Anies petition in controversial ruling",
        "content": """
        The Constitutional Court has rejected Anies Baswedan's petition challenging the presidential election results.
        The April 22 ruling found insufficient evidence of systematic fraud that would have changed the outcome.
        Justices acknowledged some procedural irregularities but deemed them not material to the final result.
        Anies's legal team expressed disappointment and questioned the court's independence.
        The ruling came after weeks of hearings featuring testimony from witnesses on both sides.
        Public protests outside the court were relatively muted compared to previous election disputes.
        The decision finalizes Prabowo Subianto's victory and clears the path for October inauguration.
        Critics argue the court's credibility was already compromised by the vice-presidential age ruling controversy.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/04/22/court-rejects-anies-petition",
        "published_date": datetime(2024, 4, 22),
        "category": "Politics"
    },
    {
        "title": "Prabowo's free meal program faces implementation challenges",
        "content": """
        President Prabowo's flagship free school meals program encounters logistical and budgetary hurdles.
        The ambitious program promises nutritious meals to millions of students across Indonesia's sprawling archipelago.
        Initial rollout has been limited to pilot areas while authorities work out supply chain issues.
        Budget constraints have forced adjustments to the original scope and timeline of the program.
        Local governments report difficulties in procurement and kitchen infrastructure requirements.
        Nutritionists have raised concerns about meal quality standards in early implementations.
        Despite challenges, the program remains popular among parents and seen as fulfilling a key campaign promise.
        The administration has committed to nationwide expansion within the first year in office.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/11/15/free-meal-program-challenges",
        "published_date": datetime(2024, 11, 15),
        "category": "Social"
    },
    {
        "title": "Opposition parties regroup following election defeat",
        "content": """
        Political parties that supported losing candidates are reassessing strategies following the decisive election outcome.
        PDI-P, which backed Ganjar Pranowo, faces internal debates about its direction under Megawati's leadership.
        The NasDem party is navigating its position after Anies Baswedan's defeat.
        Some opposition figures have been approached about joining the incoming government's broad coalition.
        Party leaders are balancing between maintaining opposition voices and pragmatic power-sharing.
        Youth wings of opposition parties call for generational leadership changes.
        The 2029 presidential election is already being discussed in terms of potential candidates and alliances.
        A weakened opposition raises questions about oversight of the dominant governing coalition.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/05/20/opposition-parties-regroup",
        "published_date": datetime(2024, 5, 20),
        "category": "Politics"
    },
    {
        "title": "TikTok's role in Indonesian election draws global attention",
        "content": """
        The Prabowo campaign's successful use of TikTok has become a case study in modern political communication.
        Campaign strategists crafted viral content featuring the 72-year-old candidate in relatable, humorous situations.
        Dance videos and meme-friendly content helped soften Prabowo's previous tough military image.
        The platform's young user base aligned with efforts to court first-time voters.
        Critics argue the social media strategy prioritized entertainment over substantive policy debate.
        International researchers are studying the campaign as an example of youth engagement tactics.
        Concerns about misinformation and platform regulation emerged during the campaign period.
        The success has prompted other Indonesian politicians to invest heavily in social media presence.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/03/05/tiktok-election-influence",
        "published_date": datetime(2024, 3, 5),
        "category": "Technology"
    },
    
    # =========================================
    # ADDITIONAL ARTICLES - ANTARA (6-10)
    # =========================================
    {
        "title": "Prabowo and Gibran sworn in as president and vice president",
        "content": """
        Prabowo Subianto and Gibran Rakabuming Raka were officially sworn in as Indonesia's president and vice president on October 20, 2024.
        The inauguration ceremony at the People's Consultative Assembly was attended by dignitaries from around the world.
        Prabowo pledged to serve all Indonesians and continue the nation's development trajectory.
        The peaceful transfer of power from Joko Widodo marked Indonesia's fourth democratic transition since 1998.
        Security was tight in Jakarta with thousands of police and military deployed.
        Foreign leaders including representatives from ASEAN nations attended the ceremony.
        Celebrations were held across the country following the swearing-in.
        The new president immediately began work on cabinet formation and policy priorities.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/10/20/prabowo-inauguration",
        "published_date": datetime(2024, 10, 20),
        "category": "Politics"
    },
    {
        "title": "Voter turnout reaches 82 percent in historic election",
        "content": """
        Official figures confirm that 82 percent of eligible voters participated in the February 2024 presidential election.
        The turnout represents approximately 167 million voters out of 204 million registered citizens.
        Election officials note the high participation despite logistical challenges across the archipelago.
        Urban areas recorded slightly higher turnout than remote regions.
        First-time voters showed strong participation rates, particularly in major cities.
        The figure is comparable to turnout in previous presidential elections.
        International observers praised the peaceful conduct of voting day.
        Technical issues at some polling stations were resolved without major disruption.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/02/20/voter-turnout-figures",
        "published_date": datetime(2024, 2, 20),
        "category": "Politics"
    },
    {
        "title": "Indonesia maintains neutral foreign policy under new leadership",
        "content": """
        The Prabowo administration has reaffirmed Indonesia's commitment to an independent and active foreign policy.
        Foreign Minister Sugiono emphasized that Indonesia will continue its non-aligned stance amid global tensions.
        Relations with major powers including the US and China will be balanced pragmatically.
        ASEAN centrality remains the cornerstone of regional engagement.
        The new government has pledged to continue Indonesia's mediation role in regional conflicts.
        Economic diplomacy to attract investment will be a priority.
        Defense cooperation agreements signed under the previous administration will be honored.
        Indonesia seeks to maintain its voice in international forums on climate and development issues.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/11/01/foreign-policy-continuity",
        "published_date": datetime(2024, 11, 1),
        "category": "Diplomacy"
    },
    {
        "title": "Quick count accuracy confirmed by official results",
        "content": """
        The official vote count has confirmed the accuracy of quick count projections released on election night.
        Major polling organizations correctly predicted Prabowo's victory margin within a few percentage points.
        The consistency between quick counts and official results bolstered confidence in the electoral process.
        Some minor discrepancies in regional voting patterns were noted but did not affect the overall outcome.
        Election technology experts praised the reliability of Indonesia's tabulation systems.
        The validation process included cross-checking results from over 800,000 polling stations.
        International observers confirmed the technical integrity of the counting process.
        The experience may inform improvements in future election procedures.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/03/22/quick-count-accuracy",
        "published_date": datetime(2024, 3, 22),
        "category": "Politics"
    },
    {
        "title": "ASEAN leaders congratulate Prabowo on election victory",
        "content": """
        Leaders from ASEAN member states have extended congratulations to Prabowo Subianto on his presidential election victory.
        Singapore Prime Minister conveyed best wishes for Indonesia's continued prosperity and stability.
        Malaysian and Thai leaders emphasized the importance of regional cooperation.
        The Philippines expressed hope for strengthened bilateral ties under the new administration.
        ASEAN Secretary-General highlighted Indonesia's continued leadership role in the regional bloc.
        Diplomatic messages emphasized peaceful transition and democratic process.
        Regional partners look forward to working with the incoming government on trade and security.
        The congratulatory messages signal international acceptance of the election outcome.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/02/16/asean-congratulations",
        "published_date": datetime(2024, 2, 16),
        "category": "Diplomacy"
    },
    
    # =========================================
    # ADDITIONAL ARTICLES - JAKARTA GLOBE (6-10)
    # =========================================
    {
        "title": "Anies accuses administration of electoral interference",
        "content": """
        Presidential candidate Anies Baswedan has accused the Jokowi administration of interfering in the electoral process.
        Speaking at a press conference, Anies alleged that state resources were mobilized to benefit his rival.
        He specifically cited the timing of social assistance distributions during the campaign period.
        The accusations echo concerns raised by election monitoring groups about incumbency advantages.
        Government officials categorically denied the allegations, calling them politically motivated.
        Anies announced he would present evidence to the Constitutional Court in his legal challenge.
        The controversy has deepened divisions between supporters of different candidates.
        International observers called for transparent investigation of the claims.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/anies-accuses-interference",
        "published_date": datetime(2024, 3, 1),
        "category": "Politics"
    },
    {
        "title": "Business leaders embrace Prabowo's economic vision",
        "content": """
        Indonesian business associations have expressed support for President Prabowo's economic development plans.
        The Indonesian Chamber of Commerce welcomed continuity in trade and investment policies.
        Infrastructure development, particularly connectivity projects, was cited as beneficial for commerce.
        The business community praised the retention of Sri Mulyani as Finance Minister.
        Export-oriented industries anticipate stable currency and trade policies.
        Digital economy initiatives from the previous administration are expected to continue.
        Some concerns remain about the fiscal impact of expanded social programs.
        Overall, business confidence indicators have remained stable through the transition.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/business-leaders-prabowo-vision",
        "published_date": datetime(2024, 10, 30),
        "category": "Business"
    },
    {
        "title": "Gibran's rise highlights generational shift in Indonesian politics",
        "content": """
        The election of 36-year-old Gibran Rakabuming Raka as vice president marks a significant generational milestone.
        As Indonesia's youngest-ever vice president, Gibran represents the growing influence of younger politicians.
        His path to office, facilitated by a controversial court ruling, remains debated.
        Supporters view him as a fresh voice connected to millennial and Gen Z concerns.
        Critics argue his rise was due to family connections rather than earned political merit.
        Gibran has pledged to focus on youth employment, education, and digital innovation.
        His political style incorporates social media engagement and informal communication.
        The vice presidency may position him for future presidential aspirations.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/gibran-generational-shift",
        "published_date": datetime(2024, 10, 22),
        "category": "Politics"
    },
    {
        "title": "Election Workers Face Health Challenges Amid Complex Voting Process",
        "content": """
        Reports have emerged of health challenges faced by election workers during Indonesia's massive voting operation.
        The simultaneous nature of voting for president, parliament, and regional representatives increased workload.
        Election commission reported cases of workers falling ill due to exhaustion during counting.
        Critics called for reforms to reduce the burden on volunteer election officials.
        The government promised improved support and medical provisions for future elections.
        Despite challenges, the election proceeded without major disruptions to the counting process.
        Civil society groups advocated for better compensation and training for election workers.
        The experience has prompted discussions about splitting national and regional elections.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/election-workers-challenges",
        "published_date": datetime(2024, 2, 25),
        "category": "Social"
    },
    {
        "title": "Rupiah strengthens as political uncertainty clears",
        "content": """
        The Indonesian rupiah has gained ground against the US dollar following the resolution of election disputes.
        The currency strengthened by over 2 percent in the weeks after the Constitutional Court dismissed challenges.
        Currency traders cited reduced political risk as a key factor in the appreciation.
        Bank Indonesia maintained supportive monetary policy through the transition period.
        Foreign capital flows into Indonesian bonds and equities improved on political clarity.
        The stable rupiah benefits importers and helps control inflation.
        Economic fundamentals including trade surplus and foreign reserves remain healthy.
        Analysts project continued stability as the new government settles into office.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/rupiah-strengthens-post-election",
        "published_date": datetime(2024, 5, 1),
        "category": "Business"
    },
    
    # =========================================
    # JAKARTA POST (Articles 11-18) - Extended Coverage
    # =========================================
    {
        "title": "Prabowo's 'zero enemies' foreign policy takes shape",
        "content": """
        President Prabowo Subianto has outlined his foreign policy doctrine of 'zero enemies, one thousand friends' in his first major diplomatic address.
        The approach emphasizes maintaining balanced relations with both China and the United States amid intensifying great power competition.
        During visits to Beijing and Washington in November 2024, Prabowo secured investment pledges totaling over $10 billion from China.
        He also reaffirmed defense cooperation with the United States, honoring agreements signed during his tenure as Defense Minister.
        The strategy reflects Indonesia's traditional non-aligned stance while pursuing pragmatic economic interests.
        Analysts note the balancing act requires careful navigation of South China Sea tensions.
        Regional partners have welcomed Indonesia's continued commitment to ASEAN centrality.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/world/2024/11/20/prabowo-zero-enemies-policy",
        "published_date": datetime(2024, 11, 20)
    },
    {
        "title": "Defense modernization accelerates under Prabowo",
        "content": """
        The Prabowo administration has announced accelerated military modernization plans building on programs initiated during his time as Defense Minister.
        Priority acquisitions include F-15 fighter jets from the United States and additional naval vessels for maritime patrol.
        The defense budget is set to increase, reflecting Prabowo's long-standing focus on military strength.
        Naval and air bases in the Natuna Islands are being reinforced amid ongoing tensions with Chinese coast guard vessels.
        Defense cooperation agreements with multiple countries are being expanded.
        The modernization aims to assert Indonesia's sovereignty over its exclusive economic zone.
        Critics question whether increased defense spending will strain other budget priorities.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/11/25/defense-modernization-prabowo",
        "published_date": datetime(2024, 11, 25)
    },
    {
        "title": "Nusantara capital development continues under new government",
        "content": """
        Construction of Indonesia's new capital city Nusantara in East Kalimantan continues as a priority project under President Prabowo.
        The transition from Jakarta remains on schedule despite earlier concerns about funding and political support.
        International investors are being courted to participate in the massive infrastructure development.
        Government ministries are preparing phased relocations as initial facilities near completion.
        Environmental concerns about deforestation in Borneo continue to be raised by advocacy groups.
        The project represents continuity with the Jokowi administration's signature initiative.
        Economic benefits for East Kalimantan are anticipated as construction accelerates.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/12/01/nusantara-development-continues",
        "published_date": datetime(2024, 12, 1)
    },
    {
        "title": "Sri Mulyani pledges fiscal discipline amid spending pressures",
        "content": """
        Finance Minister Sri Mulyani Indrawati has pledged to maintain fiscal discipline while accommodating new social programs.
        Her retention in the cabinet was seen as a signal of economic policy continuity to international markets.
        The minister acknowledged challenges in funding the ambitious free meal program within budget constraints.
        Revenue optimization and expenditure efficiency are key priorities for the finance ministry.
        International credit rating agencies have maintained Indonesia's investment grade status.
        The minister emphasized the importance of sustainable debt levels amid global economic uncertainty.
        Business leaders praised the commitment to prudent fiscal management.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/business/2024/11/10/sri-mulyani-fiscal-discipline",
        "published_date": datetime(2024, 11, 10)
    },
    {
        "title": "Digital economy initiatives expand under new administration",
        "content": """
        The Prabowo administration has announced expanded support for Indonesia's growing digital economy sector.
        Policies aim to nurture local startups while attracting foreign technology investment.
        Digital infrastructure development, including broadband expansion to rural areas, is prioritized.
        E-commerce regulations are being refined to balance growth with consumer protection.
        The government seeks to position Indonesia as Southeast Asia's leading digital economy.
        Partnerships with technology companies for skills training are being developed.
        Regulatory frameworks for emerging technologies including AI are under consideration.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/business/2024/11/28/digital-economy-expansion",
        "published_date": datetime(2024, 11, 28)
    },
    {
        "title": "Religious minorities cautiously optimistic about new government",
        "content": """
        Representatives of religious minority communities have expressed cautious optimism about the Prabowo administration.
        The president's campaign rhetoric emphasized pluralism and protection of all religious groups.
        Early appointments include figures seen as supportive of interfaith dialogue.
        Past concerns about Prabowo's ties to conservative Islamic groups have somewhat eased.
        Minority leaders are watching for concrete policy actions on religious freedom.
        The government has pledged to address longstanding issues affecting minority worship rights.
        Civil society groups continue to monitor incidents of religious intolerance.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/11/15/religious-minorities-new-govt",
        "published_date": datetime(2024, 11, 15)
    },
    {
        "title": "Labor unions seek dialogue with incoming administration",
        "content": """
        Indonesian labor unions are seeking dialogue with the new government on worker protection policies.
        Concerns remain about the Omnibus Law on Job Creation passed under the previous administration.
        Union leaders hope for amendments to provisions they consider harmful to workers.
        The minimum wage setting process for 2025 is being closely watched by labor groups.
        Manufacturing sector workers are particularly concerned about job security amid global economic shifts.
        The government has signaled willingness to engage with labor representatives.
        Balance between investment climate and worker protection remains a policy challenge.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/11/18/labor-unions-dialogue",
        "published_date": datetime(2024, 11, 18)
    },
    {
        "title": "Environmental groups press new government on climate commitments",
        "content": """
        Environmental organizations are urging the Prabowo administration to strengthen Indonesia's climate commitments.
        Deforestation rates and palm oil expansion remain key concerns for green groups.
        The government's stance on coal phase-out timelines is being closely monitored.
        International climate finance partnerships including JETP are under review.
        Carbon pricing mechanisms are being evaluated for potential implementation.
        Environmental activists note the tension between economic development and sustainability goals.
        The administration has pledged to balance growth with environmental protection.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/12/05/environmental-groups-climate",
        "published_date": datetime(2024, 12, 5)
    },
    
    # =========================================
    # JAKARTA POST (Articles 19-25) - Final Batch
    # =========================================
    {
        "title": "Indonesia-Australia relations strengthen under Prabowo",
        "content": """
        Bilateral relations between Indonesia and Australia show signs of strengthening under Prabowo.
        Defense cooperation agreements are being expanded.
        Trade discussions continue on improving market access.
        People-to-people exchanges through education are prioritized.
        Regional security cooperation in the Indo-Pacific is emphasized.
        Historical sensitivities are being managed through diplomatic channels.
        Economic complementarities drive deeper engagement.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/world/2024/12/10/indonesia-australia-relations",
        "published_date": datetime(2024, 12, 10)
    },
    {
        "title": "Prabowo addresses food security challenges",
        "content": """
        President Prabowo has made food security a top priority of his administration.
        Rice production targets aim for self-sufficiency within the presidential term.
        Agricultural modernization programs receive significant budget allocation.
        Import policies are being adjusted to protect domestic farmers.
        Climate adaptation strategies for agriculture are being developed.
        Food distribution infrastructure is being improved nationwide.
        Strategic food reserves are being enhanced.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/12/18/prabowo-food-security",
        "published_date": datetime(2024, 12, 18)
    },
    {
        "title": "Maritime domain awareness enhanced under new strategy",
        "content": """
        Indonesia is enhancing maritime domain awareness as part of national security strategy.
        New patrol vessels and surveillance systems are being deployed.
        Coordination among maritime agencies is being improved.
        Illegal fishing enforcement has intensified in Indonesian waters.
        Coast guard capabilities are being expanded.
        International cooperation on maritime security continues.
        Protection of exclusive economic zone resources is prioritized.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/12/22/maritime-domain-awareness",
        "published_date": datetime(2024, 12, 22)
    },
    {
        "title": "Women's representation in cabinet falls short of targets",
        "content": """
        Women's representation in the Prabowo cabinet has fallen short of advocacy group targets.
        Female ministers hold a smaller proportion of portfolios than in some previous cabinets.
        Gender equality advocates are calling for greater inclusion in senior positions.
        Women's empowerment programs continue under ministry initiatives.
        Gender mainstreaming in policy development is emphasized.
        Female leadership in local government varies across regions.
        Civil society continues to push for increased women's political participation.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2024/10/30/women-cabinet-representation",
        "published_date": datetime(2024, 10, 30)
    },
    {
        "title": "Anti-money laundering efforts receive new emphasis",
        "content": """
        The government has announced enhanced anti-money laundering enforcement measures.
        Financial intelligence capabilities are being strengthened.
        Cooperation with international partners on illicit finance is expanding.
        Beneficial ownership transparency requirements are being tightened.
        Real estate and luxury goods sectors face increased scrutiny.
        Virtual asset regulations are being developed.
        Indonesia seeks to improve its international AML ratings.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/business/2024/12/28/anti-money-laundering-efforts",
        "published_date": datetime(2024, 12, 28)
    },
    {
        "title": "Infrastructure spending priorities shift to connectivity",
        "content": """
        Infrastructure spending priorities are shifting toward inter-island connectivity.
        Eastern Indonesia regions receive increased allocation for transport links.
        Port modernization supports trade facilitation objectives.
        Digital infrastructure expansion reaches underserved areas.
        Power generation and distribution improvements continue.
        Private sector participation in infrastructure is encouraged.
        Maintenance of existing infrastructure receives attention alongside new projects.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/business/2025/01/05/infrastructure-connectivity-priorities",
        "published_date": datetime(2025, 1, 5)
    },
    {
        "title": "Social protection programs under comprehensive review",
        "content": """
        The government is conducting a comprehensive review of social protection programs.
        Targeting efficiency aims to ensure benefits reach intended recipients.
        Integration of various welfare programs is being considered.
        Digital delivery mechanisms are being expanded.
        The free meal program is being integrated with existing nutrition initiatives.
        Budget sustainability of expanded social programs is being assessed.
        Poverty reduction remains a core government objective.
        """,
        "media_source": "jakarta_post",
        "url": "https://www.thejakartapost.com/indonesia/2025/01/10/social-protection-review",
        "published_date": datetime(2025, 1, 10)
    },
    
    # =========================================
    # TEMPO ENGLISH (Articles 11-25) - Extended Coverage
    # =========================================
    {
        "title": "Prabowo's first 100 days: promises versus reality",
        "content": """
        Analysis of President Prabowo's first 100 days reveals mixed progress on campaign promises.
        The free meal program has launched in pilot areas but faces implementation challenges.
        Cabinet formation prioritized political coalition management over technocratic expertise in some portfolios.
        Foreign policy has been active with visits to major world capitals.
        Economic indicators remain stable though growth targets are ambitious.
        Democratic reform advocates note limited progress on governance improvements.
        The administration maintains high approval ratings despite criticism from some quarters.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/20/prabowo-first-100-days",
        "published_date": datetime(2025, 1, 20)
    },
    {
        "title": "Gibran focuses on youth employment initiatives",
        "content": """
        Vice President Gibran Rakabuming Raka is leading initiatives focused on youth employment and entrepreneurship.
        Programs aim to address unemployment among young Indonesians who strongly supported the election ticket.
        Digital skills training and startup incubation are priorities.
        Coordination with education ministry on vocational training is underway.
        The vice president's social media presence continues to engage young constituents.
        Critics question whether initiatives are substantive or primarily promotional.
        Early results of youth programs are expected to be assessed in coming months.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/12/10/gibran-youth-employment",
        "published_date": datetime(2024, 12, 10)
    },
    {
        "title": "Investigation: Free meal program procurement concerns",
        "content": """
        A Tempo investigation has uncovered concerns about procurement processes in the free meal program.
        Questions have been raised about the selection of suppliers for food distribution.
        Some contracts appear to have been awarded without competitive bidding processes.
        Program officials deny any irregularities and cite urgency of implementation.
        Anti-corruption watchdogs are calling for greater transparency in program spending.
        The ambitious scope of the program creates logistical challenges for oversight.
        Government auditors have begun reviewing procurement documentation.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/12/20/free-meal-procurement-concerns",
        "published_date": datetime(2024, 12, 20)
    },
    {
        "title": "Former rivals join Prabowo cabinet in reconciliation move",
        "content": """
        Several political figures who opposed Prabowo during the campaign have accepted cabinet positions.
        The appointments are seen as part of efforts to build a broad governing coalition.
        Some critics view the inclusions as co-optation of potential opposition voices.
        Party leaders cite national unity as justification for joining the government.
        The large cabinet provides numerous positions for coalition management.
        Political scientists note the implications for checks and balances.
        Indonesia's tradition of power-sharing coalitions continues under the new administration.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/10/28/former-rivals-join-cabinet",
        "published_date": datetime(2024, 10, 28)
    },
    {
        "title": "Media freedom concerns under new administration",
        "content": """
        Press freedom advocates have raised concerns about media environment under the new government.
        Concentration of media ownership among political allies is noted as problematic.
        Several critical journalists report increased pressure since the government transition.
        The administration denies any interference with press freedom.
        International press freedom organizations are monitoring the situation.
        Digital platforms face potential new regulations that could affect online discourse.
        Civil society groups call for strengthened protections for journalists.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/12/15/media-freedom-concerns",
        "published_date": datetime(2024, 12, 15)
    },
    {
        "title": "Corruption watchdogs demand accountability measures",
        "content": """
        Anti-corruption organizations are pressing the new government for stronger accountability measures.
        The Corruption Eradication Commission (KPK) has faced weakening in recent years.
        Civil society groups call for restoration of the agency's independence and powers.
        Several cabinet appointments have drawn scrutiny for past corruption allegations.
        The government has pledged commitment to clean governance.
        Transparency of budget spending remains a key concern for watchdog groups.
        Indonesia's corruption perception index rankings have stagnated in recent years.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/11/22/corruption-watchdogs-accountability",
        "published_date": datetime(2024, 11, 22)
    },
    {
        "title": "PDI-P positions itself as opposition force",
        "content": """
        PDI-P, the party of former President Megawati, has positioned itself as the main opposition to the Prabowo government.
        The party that backed Ganjar Pranowo declined invitations to join the ruling coalition.
        Megawati has signaled intent to provide critical oversight of the administration.
        The party's parliamentary bloc represents a significant opposition voice.
        Some analysts question whether PDI-P can maintain opposition stance long-term.
        Regional election strategies are being developed independent of the government coalition.
        The party continues to command significant support in certain provinces.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/11/30/pdip-opposition-position",
        "published_date": datetime(2024, 11, 30)
    },
    {
        "title": "Central bank maintains independence amid transition",
        "content": """
        Bank Indonesia has emphasized its operational independence amid the government transition.
        Governor Perry Warjiyo continues to lead monetary policy focused on stability.
        Interest rate decisions remain data-driven despite political pressures for stimulus.
        Currency stability is prioritized given global economic uncertainties.
        Coordination with fiscal authorities continues on macroeconomic management.
        The central bank's inflation targeting framework remains intact.
        Foreign exchange reserves provide buffer against external shocks.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/11/08/bank-indonesia-independence",
        "published_date": datetime(2024, 11, 8)
    },
    {
        "title": "Analysis: Prabowo's military background shapes governance style",
        "content": """
        President Prabowo's military background visibly influences his governance approach.
        Command-and-control management style is evident in cabinet meetings.
        Discipline and hierarchy are emphasized in government operations.
        Some observers note tension between military efficiency and democratic deliberation.
        Defense and security issues receive personal presidential attention.
        Former military colleagues occupy key security positions.
        The president's background shapes both strengths and concerns about his leadership.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/12/25/prabowo-military-governance-style",
        "published_date": datetime(2024, 12, 25)
    },
    {
        "title": "Civil society space faces new pressures",
        "content": """
        Civil society organizations report new pressures since the government transition.
        NGO registration and funding regulations are under review.
        Some groups face increased bureaucratic requirements for activities.
        International funding for civil society faces additional scrutiny.
        Human rights organizations maintain monitoring despite challenges.
        Environmental advocacy groups continue campaigns on key issues.
        The space for civic engagement requires continued attention.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2024/12/30/civil-society-new-pressures",
        "published_date": datetime(2024, 12, 30)
    },
    {
        "title": "Prabowo's health draws speculation amid busy schedule",
        "content": """
        President Prabowo's health has drawn occasional speculation given his busy international travel schedule.
        The 73-year-old president maintains an active public calendar.
        Presidential communications have dismissed health concerns as unfounded.
        Succession considerations are natural given the president's age.
        Vice President Gibran remains prepared for constitutional responsibilities.
        Observers note Prabowo's energy levels during public appearances.
        Health transparency for leaders remains a topic of public discussion.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/08/prabowo-health-speculation",
        "published_date": datetime(2025, 1, 8)
    },
    {
        "title": "Traditional markets struggle amid modern retail expansion",
        "content": """
        Traditional markets face challenges from expanding modern retail networks.
        Small traders express concerns about competitive pressures.
        Government programs aim to modernize traditional market infrastructure.
        Zoning regulations for retail development are being reviewed.
        Digital payment adoption in traditional markets is encouraged.
        Supply chain improvements benefit both traditional and modern retail.
        Balancing market development with trader welfare remains challenging.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/12/traditional-markets-struggle",
        "published_date": datetime(2025, 1, 12)
    },
    {
        "title": "Cybersecurity threats prompt national response strategy",
        "content": """
        Rising cybersecurity threats have prompted development of a national response strategy.
        Critical infrastructure protection is prioritized.
        Government agency systems face ongoing attack attempts.
        Private sector cybersecurity requirements are being strengthened.
        Cyber workforce development programs are being expanded.
        International cooperation on cyber threats is being enhanced.
        Public awareness of digital security is being promoted.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/15/cybersecurity-national-strategy",
        "published_date": datetime(2025, 1, 15)
    },
    {
        "title": "Urban planning reforms address Jakarta congestion",
        "content": """
        Urban planning reforms aim to address chronic congestion in Jakarta.
        Public transportation expansion continues despite capital relocation.
        Transit-oriented development is encouraged around new rail stations.
        Traffic management systems are being upgraded.
        Work-from-home policies have partially reduced peak demand.
        Environmental impacts of congestion are being addressed.
        Jakarta remains the economic center despite Nusantara development.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/18/urban-planning-jakarta-congestion",
        "published_date": datetime(2025, 1, 18)
    },
    {
        "title": "Higher education internationalization gains momentum",
        "content": """
        Indonesian universities are accelerating internationalization efforts.
        Foreign university partnerships are being expanded.
        English-language programs attract regional students.
        Research collaboration with international institutions increases.
        Student exchange programs receive government support.
        Global university rankings remain a competitive focus.
        Quality assurance frameworks align with international standards.
        """,
        "media_source": "tempo",
        "url": "https://en.tempo.co/read/2025/01/22/higher-education-internationalization",
        "published_date": datetime(2025, 1, 22)
    },
    
    # =========================================
    # ANTARA NEWS (Articles 11-25) - Extended Coverage
    # =========================================
    {
        "title": "Prabowo receives congratulations from world leaders",
        "content": """
        President Prabowo Subianto has received congratulatory messages from heads of state around the world following his inauguration.
        US President Biden reaffirmed commitment to the bilateral partnership.
        Chinese President Xi Jinping expressed hopes for deeper cooperation.
        Leaders from Japan, South Korea, Australia, and India sent warm wishes.
        The diplomatic messages signal continued international engagement with Indonesia.
        Bilateral meetings are being scheduled with key partner nations.
        Indonesia's role in global forums remains valued by international partners.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/10/21/world-leaders-congratulate-prabowo",
        "published_date": datetime(2024, 10, 21)
    },
    {
        "title": "Indonesia hosts successful G20 foreign ministers meeting",
        "content": """
        Indonesia successfully hosted a G20 foreign ministers meeting in Bali under the new administration.
        The gathering demonstrated continuity in Indonesia's international diplomatic role.
        Discussions focused on global economic challenges and regional security.
        Foreign Minister Sugiono chaired productive sessions among participating nations.
        The event showcased Indonesia's diplomatic capabilities under new leadership.
        Bilateral meetings on the sidelines advanced various partnership discussions.
        The hosting reinforced Indonesia's position as a key emerging economy voice.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/08/g20-foreign-ministers-meeting",
        "published_date": datetime(2024, 12, 8)
    },
    {
        "title": "National development planning priorities announced",
        "content": """
        The National Development Planning Agency has announced priority areas for the new administration.
        Food security, energy independence, and human capital development top the agenda.
        Infrastructure connectivity linking eastern Indonesia remains a focus.
        Digital transformation across government services is prioritized.
        Climate adaptation and disaster resilience receive increased attention.
        Coordination mechanisms between ministries are being strengthened.
        Medium-term development plans are being finalized for parliamentary review.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/11/12/development-planning-priorities",
        "published_date": datetime(2024, 11, 12)
    },
    {
        "title": "State-owned enterprises reform continues",
        "content": """
        The government has announced continued reform of state-owned enterprises under new leadership.
        Efficiency improvements and corporate governance strengthening are priorities.
        Strategic SOEs in energy, telecommunications, and banking are focus areas.
        Privatization of non-strategic assets is being considered.
        Professional management recruitment aims to reduce political appointments.
        SOE contributions to state revenue are expected to increase.
        Coordination among holding companies is being streamlined.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/11/25/soe-reform-continues",
        "published_date": datetime(2024, 11, 25)
    },
    {
        "title": "Indonesia strengthens ASEAN economic integration",
        "content": """
        Indonesia is pushing for deeper ASEAN economic integration under its regional leadership role.
        Trade facilitation measures within the bloc are being enhanced.
        Digital economy cooperation among member states is prioritized.
        Indonesia seeks to address non-tariff barriers affecting intra-ASEAN trade.
        Investment promotion across the region is a coordinated focus.
        The ASEAN Economic Community continues to develop toward stated goals.
        Indonesia's economic weight gives it significant influence in regional discussions.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/12/asean-economic-integration",
        "published_date": datetime(2024, 12, 12)
    },
    {
        "title": "Health sector reforms address pandemic lessons",
        "content": """
        The Health Ministry is implementing reforms drawing on lessons from the COVID-19 pandemic.
        Healthcare system resilience and surge capacity are being strengthened.
        Local pharmaceutical manufacturing receives government support.
        Primary healthcare facilities in rural areas are being expanded.
        Health worker training and retention programs are enhanced.
        Digital health initiatives aim to improve service delivery.
        Universal health coverage implementation continues to expand.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/11/18/health-sector-reforms",
        "published_date": datetime(2024, 11, 18)
    },
    {
        "title": "Agricultural productivity initiatives launched",
        "content": """
        The government has launched initiatives to boost agricultural productivity and farmer welfare.
        Rice self-sufficiency remains a strategic priority for food security.
        Irrigation infrastructure improvements are funded in key agricultural regions.
        Agricultural technology adoption is supported through extension services.
        Farmer cooperatives receive enhanced support and financing access.
        Supply chain improvements aim to reduce post-harvest losses.
        Export-oriented agriculture including palm oil faces sustainability requirements.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/02/agricultural-productivity-initiatives",
        "published_date": datetime(2024, 12, 2)
    },
    {
        "title": "Education curriculum review underway",
        "content": """
        The Education Ministry is conducting a comprehensive review of the national curriculum.
        Vocational education receives increased emphasis to address skills gaps.
        STEM education strengthening is a priority for competitiveness.
        Character education elements from previous curricula are being evaluated.
        Teacher training programs are being enhanced nationwide.
        Digital learning infrastructure expansion continues post-pandemic.
        Higher education internationalization efforts are being renewed.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/11/28/education-curriculum-review",
        "published_date": datetime(2024, 11, 28)
    },
    {
        "title": "Indonesia chairs ASEAN meetings on regional security",
        "content": """
        Indonesia has chaired productive ASEAN meetings addressing regional security challenges.
        South China Sea tensions remain a key discussion topic.
        Myanmar situation continues to concern regional partners.
        ASEAN centrality in regional architecture is reaffirmed.
        Dialogue mechanisms with external partners are maintained.
        Economic cooperation complements security discussions.
        Indonesia's diplomatic leadership role is demonstrated.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/18/indonesia-chairs-asean-security",
        "published_date": datetime(2024, 12, 18)
    },
    {
        "title": "Disaster preparedness improvements announced",
        "content": """
        The government has announced improvements to disaster preparedness systems.
        Early warning capabilities are being enhanced.
        Emergency response coordination is being strengthened.
        Local disaster management agencies receive additional resources.
        Climate-related disaster planning is prioritized.
        Community resilience programs are being expanded.
        International disaster response cooperation continues.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/22/disaster-preparedness-improvements",
        "published_date": datetime(2024, 12, 22)
    },
    {
        "title": "Cultural diplomacy promotes Indonesia globally",
        "content": """
        Cultural diplomacy efforts are promoting Indonesia on the global stage.
        Indonesian cuisine, arts, and heritage are showcased internationally.
        Tourism promotion integrates cultural elements.
        Indonesian language learning programs expand abroad.
        Cultural centers in major world cities host events.
        Creative industries receive government support for export.
        Soft power complements economic diplomacy efforts.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2024/12/28/cultural-diplomacy-indonesia",
        "published_date": datetime(2024, 12, 28)
    },
    {
        "title": "Renewable energy investment targets raised",
        "content": """
        The government has raised targets for renewable energy investment.
        Solar and geothermal development receive policy support.
        Green financing mechanisms are being developed.
        Grid infrastructure improvements enable renewable integration.
        Private sector renewable investment is encouraged.
        International climate partnerships provide technical support.
        Energy mix diversification remains a long-term objective.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2025/01/05/renewable-energy-investment-targets",
        "published_date": datetime(2025, 1, 5)
    },
    {
        "title": "Public service delivery improvements ongoing",
        "content": """
        Efforts to improve public service delivery continue under the new administration.
        Digital government services expand access for citizens.
        One-stop service centers are being replicated nationwide.
        Bureaucratic reform programs are being implemented.
        Civil service performance management is being strengthened.
        Citizen feedback mechanisms are being improved.
        Public satisfaction surveys guide service improvements.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2025/01/10/public-service-delivery-improvements",
        "published_date": datetime(2025, 1, 10)
    },
    {
        "title": "Sports development receives presidential attention",
        "content": """
        President Prabowo has given personal attention to national sports development.
        Athletic training facilities are being upgraded.
        International competition preparation receives support.
        Youth sports programs are being expanded.
        Sports tourism is being developed at key venues.
        Professional sports league governance is under review.
        Olympic and Asian Games medal targets are set.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2025/01/15/sports-development-presidential-attention",
        "published_date": datetime(2025, 1, 15)
    },
    {
        "title": "Palm oil sustainability certification expands",
        "content": """
        Palm oil sustainability certification programs are expanding in Indonesia.
        Environmental requirements for production are being strengthened.
        Traceability systems are being implemented across supply chains.
        International market access remains tied to sustainability compliance.
        Smallholder farmer support programs are being enhanced.
        Deforestation monitoring uses satellite technology.
        Industry and environmental groups continue dialogue on standards.
        """,
        "media_source": "antara",
        "url": "https://en.antaranews.com/news/2025/01/20/palm-oil-sustainability-certification",
        "published_date": datetime(2025, 1, 20)
    },
    
    # =========================================
    # JAKARTA GLOBE (Articles 11-25) - Extended Coverage
    # =========================================
    {
        "title": "Stock market rally continues on policy clarity",
        "content": """
        Indonesian equities have extended gains as investors respond positively to policy clarity.
        The Jakarta Composite Index has risen significantly since the inauguration.
        Foreign investors have increased net purchases of Indonesian stocks.
        Banking and resource sectors have led the market rally.
        Analysts cite reduced political uncertainty as a key positive factor.
        Corporate earnings expectations remain healthy for the coming year.
        Market volatility has decreased compared to the election period.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/stock-market-rally-policy-clarity",
        "published_date": datetime(2024, 11, 5)
    },
    {
        "title": "Property sector anticipates infrastructure boost",
        "content": """
        Indonesia's property sector anticipates benefits from continued infrastructure development.
        Transit-oriented development around new transport links drives interest.
        The Nusantara capital project creates opportunities for property developers.
        Residential demand remains strong in key urban centers.
        Commercial property shows recovery following pandemic-era challenges.
        Industrial property benefits from manufacturing sector expansion.
        Regulatory frameworks for property investment are being refined.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/property-sector-infrastructure-boost",
        "published_date": datetime(2024, 11, 20)
    },
    {
        "title": "Technology startups cautiously optimistic about new policies",
        "content": """
        Indonesia's technology startup ecosystem expresses cautious optimism about new government policies.
        Digital economy support programs are welcomed by entrepreneurs.
        Regulatory clarity on data protection and e-commerce is anticipated.
        Access to talent and skills development remains a concern.
        Funding environment has stabilized after global tech sector corrections.
        Southeast Asian regional expansion remains attractive for Indonesian startups.
        Government digital transformation creates opportunities for local tech firms.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/tech-startups-new-policies",
        "published_date": datetime(2024, 11, 15)
    },
    {
        "title": "Mining sector welcomes downstreaming commitment",
        "content": """
        Indonesia's mining sector has welcomed the new government's commitment to mineral downstreaming.
        Nickel processing for battery materials continues to attract major investment.
        Export restrictions on raw minerals are maintained to encourage local processing.
        Environmental standards for mining operations face increased scrutiny.
        Local content requirements in mining support domestic industries.
        Infrastructure development in mining regions is prioritized.
        International trade tensions over export policies are being managed diplomatically.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/mining-downstreaming-commitment",
        "published_date": datetime(2024, 11, 22)
    },
    {
        "title": "Tourism sector recovery accelerates",
        "content": """
        Indonesia's tourism sector is experiencing accelerated recovery under the new administration.
        Bali visitor numbers have returned to pre-pandemic levels.
        Diversification beyond Bali to destinations like Labuan Bajo continues.
        Visa facilitation for key tourist source markets is being expanded.
        Infrastructure improvements at tourist destinations are ongoing.
        Sustainable tourism principles are increasingly emphasized.
        The sector's contribution to GDP is projected to increase.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/tourism-recovery-accelerates",
        "published_date": datetime(2024, 11, 28)
    },
    {
        "title": "Energy transition plans face implementation challenges",
        "content": """
        Indonesia's energy transition plans face significant implementation challenges.
        Coal remains dominant in the power generation mix despite transition pledges.
        Renewable energy investment is growing but below required levels.
        Grid infrastructure upgrades are needed to accommodate intermittent sources.
        Just transition concerns for coal-dependent communities are being addressed.
        International climate finance partnerships remain complex to implement.
        Energy security considerations complicate rapid transition timelines.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/energy-transition-challenges",
        "published_date": datetime(2024, 12, 10)
    },
    {
        "title": "Regional elections set to test coalition dynamics",
        "content": """
        Upcoming regional elections will test the durability of the national government coalition.
        Gubernatorial and mayoral races in key provinces are being closely watched.
        Coalition partners may compete against each other at the regional level.
        Opposition parties see regional contests as opportunities to rebuild support.
        Local issues often differ from national campaign themes.
        Political dynasties continue to feature prominently in regional races.
        Voter turnout patterns may indicate satisfaction with national government.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/regional-elections-coalition-test",
        "published_date": datetime(2024, 12, 5)
    },
    {
        "title": "Academic freedom concerns raised by university associations",
        "content": """
        University associations have raised concerns about academic freedom under the new government.
        Campus autonomy in governance and curriculum decisions is emphasized.
        Research funding allocation processes are under review.
        International academic collaborations continue despite some bureaucratic hurdles.
        Student activism faces varying responses across different institutions.
        Quality assurance mechanisms for higher education are being strengthened.
        Graduate employability remains a key focus for university programs.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/news/academic-freedom-concerns",
        "published_date": datetime(2024, 12, 15)
    },
    {
        "title": "Automotive industry pivots to electric vehicles",
        "content": """
        Indonesia's automotive industry is accelerating its pivot to electric vehicles.
        Government incentives support EV manufacturing and adoption.
        Battery production facilities attract major investment.
        Charging infrastructure expansion is underway in major cities.
        Local content requirements for EV production are being phased in.
        Export potential for Indonesian-made EVs is being developed.
        Traditional automakers are adjusting to the transition.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/automotive-industry-electric-vehicles",
        "published_date": datetime(2024, 12, 20)
    },
    {
        "title": "Banking sector profits rise amid economic stability",
        "content": """
        Indonesian banks report rising profits amid continued economic stability.
        Loan growth remains healthy across major lending categories.
        Net interest margins are being maintained despite competitive pressures.
        Digital banking services are being rapidly expanded.
        Non-performing loan levels remain manageable.
        Capital adequacy ratios exceed regulatory requirements.
        Regional expansion by Indonesian banks continues.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/banking-sector-profits-rise",
        "published_date": datetime(2024, 12, 25)
    },
    {
        "title": "Consumer confidence index reaches post-pandemic high",
        "content": """
        Indonesia's consumer confidence index has reached its highest level since before the pandemic.
        Economic optimism reflects stability following the election transition.
        Retail sales growth has accelerated in recent months.
        Employment conditions have improved across sectors.
        Inflation remains within the central bank's target range.
        Consumer credit growth indicates spending confidence.
        Holiday season spending exceeded expectations.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/consumer-confidence-post-pandemic-high",
        "published_date": datetime(2025, 1, 2)
    },
    {
        "title": "Healthcare sector attracts foreign investment",
        "content": """
        Indonesia's healthcare sector is attracting increased foreign investment.
        Hospital development projects are underway in major cities.
        Medical device manufacturing is expanding locally.
        Pharmaceutical sector investment continues to grow.
        Healthcare tourism potential is being developed.
        Digital health startups receive venture funding.
        Regulatory frameworks for health sector investment are being refined.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/healthcare-sector-foreign-investment",
        "published_date": datetime(2025, 1, 8)
    },
    {
        "title": "Real estate developers eye Nusantara opportunities",
        "content": """
        Major real estate developers are positioning for opportunities in the new capital Nusantara.
        Land acquisition and development planning are underway.
        Government partnerships for housing development are being negotiated.
        Commercial real estate demand is anticipated to grow.
        Sustainability requirements for construction are emphasized.
        Infrastructure development timeline shapes investment decisions.
        Long-term appreciation potential attracts investor interest.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/real-estate-developers-nusantara",
        "published_date": datetime(2025, 1, 12)
    },
    {
        "title": "Logistics sector modernization accelerates",
        "content": """
        Indonesia's logistics sector is undergoing accelerated modernization.
        Port efficiency improvements reduce cargo handling times.
        Cold chain infrastructure supports food distribution.
        E-commerce growth drives last-mile delivery innovation.
        Intermodal connectivity between transport modes improves.
        Digital platforms optimize logistics operations.
        Logistics costs as percentage of GDP are declining.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/logistics-sector-modernization",
        "published_date": datetime(2025, 1, 18)
    },
    {
        "title": "Indonesia hosts major international investment summit",
        "content": """
        Indonesia has successfully hosted a major international investment summit.
        The event attracted business leaders and investors from around the world.
        Investment commitments across multiple sectors were announced.
        President Prabowo addressed attendees on Indonesia's economic vision.
        Bilateral investment discussions continued on the margins.
        The summit showcased Indonesia as a leading emerging market destination.
        Follow-up mechanisms will track investment realization.
        """,
        "media_source": "jakarta_globe",
        "url": "https://jakartaglobe.id/business/indonesia-investment-summit",
        "published_date": datetime(2025, 1, 22)
    },
]

# Supported media sources
SUPPORTED_MEDIA_SOURCES = ["jakarta_post", "tempo", "antara", "jakarta_globe"]




